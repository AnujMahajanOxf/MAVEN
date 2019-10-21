import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class ActorCriticLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.params = self.agent_params


        if args.critic_q_fn == "coma":
            self.critic = COMACritic(scheme, args)
        elif args.critic_q_fn == "centralV":
            self.critic = CentralVCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.params += self.critic_params
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                        eps=args.optim_eps)

        self.separate_baseline_critic = False
        if args.critic_q_fn != args.critic_baseline_fn:
            self.separate_baseline_critic = True
            if args.critic_baseline_fn == "coma":
                self.baseline_critic = COMACritic(scheme, args)
            elif args.critic_baseline_fn == "centralV":
                self.baseline_critic = CentralVCritic(scheme, args)
            self.target_baseline_critic = copy.deepcopy(self.baseline_critic)

            self.baseline_critic_params = list(self.baseline_critic.parameters())
            self.params += self.baseline_critic_params
            self.baseline_critic_optimiser = RMSprop(params=self.baseline_critic_params, lr=args.critic_lr,
                                                     alpha=args.optim_alpha,
                                                     eps=args.optim_eps)

        if args.critic_train_mode == "seq":
            self.critic_train_fn = self.train_critic_sequential
        elif args.critic_train_mode == "batch":
            self.critic_train_fn = self.train_critic_batched
        else:
            raise ValueError

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()
        baseline_critic_mask = mask.clone()

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        pi = mac_out

        for _ in range(self.args.critic_train_reps):
            q_sa, v_s, critic_train_stats = self.critic_train_fn(self.critic, self.target_critic, self.critic_optimiser, batch,
                                                                 rewards, terminated, actions, avail_actions, critic_mask)

        if self.separate_baseline_critic:
            for _ in range(self.args.critic_train_reps):
                q_sa_baseline, v_s_baseline, critic_train_stats_baseline = \
                    self.critic_train_fn(self.baseline_critic, self.target_baseline_critic, self.baseline_critic_optimiser,
                                         batch, rewards, terminated, actions, avail_actions, baseline_critic_mask)
            if self.args.critic_baseline_fn == "coma":
                baseline = (q_sa_baseline * pi).sum(-1).detach()
            else:
                baseline = v_s_baseline
        else:
            if self.args.critic_baseline_fn == "coma":
                baseline = (q_sa * pi).sum(-1).detach()
            else:
                baseline = v_s

        actions = actions[:,:-1]

        if self.critic.output_type == "q":
            q_sa = th.gather(q_sa, dim=3, index=actions).squeeze(3)
            if self.args.critic_q_fn == "coma" and self.args.coma_mean_q:
                q_sa = q_sa.mean(2, keepdim=True).expand(-1, -1, self.n_agents)
        q_sa = self.nstep_returns(rewards, mask, q_sa, self.args.q_nstep)

        advantages = (q_sa - baseline).detach().squeeze()

        # Calculate policy grad with mask

        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        pg_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, optimiser, batch, rewards, terminated, actions,
                                avail_actions, mask):
        # Optimise critic
        target_vals = target_critic(batch)

        all_vals = th.zeros_like(target_vals)

        if critic.output_type == 'q':
            target_vals = th.gather(target_vals, dim=3, index=actions)
            # target_vals = th.cat([target_vals[:, 1:], th.zeros_like(target_vals[:, 0:1])], dim=1)
        target_vals = target_vals.squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_vals, self.n_agents,
                                          self.args.gamma, self.args.td_lambda)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1) + 1)):
            vals_t = critic(batch, t)
            all_vals[:, t] = vals_t.squeeze(1)

            if t == rewards.size(1):
                continue

            mask_t = mask[:, t]
            if mask_t.sum() == 0:
                continue

            if critic.output_type == "q":
                vals_t = th.gather(vals_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            else:
                vals_t = vals_t.squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (vals_t - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()  # Not dividing by number of agents, only # valid timesteps
            optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(optimiser.param_groups[0]["params"], self.args.grad_norm_clip)
            optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((vals_t * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        if critic.output_type == 'q':
            q_vals = all_vals[:, :-1]
            v_s = None
        else:
            q_vals = all_vals[:, :-1].squeeze(3)
            v_s = all_vals[:, :-1].squeeze(3)

        return q_vals, v_s, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def train_critic_batched(self, critic, target_critic, optimiser, batch, rewards, terminated, actions,
                             avail_actions, mask):
        # Optimise critic
        target_vals = target_critic(batch)

        target_vals = target_vals[:, :-1]

        if critic.output_type == 'q':
            target_vals = th.gather(target_vals, dim=3, index=actions)
            target_vals = th.cat([target_vals[:, 1:], th.zeros_like(target_vals[:, 0:1])], dim=1)
        target_vals = target_vals.squeeze(3)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, target_vals, self.n_agents,
                                         self.args.gamma, self.args.td_lambda)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        all_vals = critic(batch)
        vals = all_vals.clone()[:, :-1]

        if critic.output_type == "q":
            vals = th.gather(vals, dim=3, index=actions)
        vals = vals.squeeze(3)

        td_error = (vals - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(optimiser.param_groups[0]["params"], self.args.grad_norm_clip)
        optimiser.step()
        self.critic_training_steps += 1

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((vals * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((targets * mask).sum().item() / mask_elems)

        if critic.output_type == 'q':
            q_vals = all_vals[:, :-1]
            v_s = None
        else:
            q_vals = build_td_lambda_targets(rewards, terminated, mask, all_vals.squeeze(3)[:, 1:], self.n_agents,
                                             self.args.gamma, self.args.td_lambda)
            v_s = vals

        return q_vals, v_s, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.separate_baseline_critic:
            self.target_baseline_critic.load_state_dict(self.baseline_critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        if self.separate_baseline_critic:
            self.baseline_critic.cuda()
            self.target_baseline_critic.cuda()
