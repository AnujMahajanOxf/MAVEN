import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qtran import QTran as QTranAlt
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        assert args.mixer == "qtran_alt"
        self.mixer = QTranAlt(args)

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl
        mac_out_maxs = mac_out.clone().detach()
        mac_out_maxs[avail_actions == 0] = -9999999

        # Best joint action computed by target agents
        cur_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
        # Best joint-action computed by regular agents
        max_actions_qvals, max_actions_current = mac_out_maxs[:, :-1].max(dim=3, keepdim=True)

        counter_qs, vs = self.mixer(batch[:, :-1])

        # Need to argmax across the target agents' actions
        # Convert cur_max_actions to one hot
        max_actions = th.zeros(size=(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, self.args.n_actions), device=batch.device)
        max_actions_onehot = max_actions.scatter(3, cur_max_actions, 1)
        max_actions_onehot_repeat = max_actions_onehot.repeat(1,1,self.args.n_agents,1)
        agent_mask = (1 - th.eye(self.args.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.args.n_actions)#.view(self.n_agents, -1)
        masked_actions = max_actions_onehot_repeat * agent_mask.unsqueeze(0).unsqueeze(0)
        masked_actions = masked_actions.view(-1, self.args.n_agents * self.args.n_actions)
        target_counter_qs, target_vs = self.target_mixer(batch[:, 1:], masked_actions)

        # Td loss
        td_target_qs = target_counter_qs.gather(1, cur_max_actions.view(-1, 1))
        td_chosen_qs = counter_qs.gather(1, actions.contiguous().view(-1, 1))
        td_targets = rewards.repeat(1,1,self.args.n_agents).view(-1, 1) + self.args.gamma * (1 - terminated.repeat(1,1,self.args.n_agents).view(-1, 1)) * td_target_qs
        td_error = (td_chosen_qs - td_targets.detach())

        td_mask = mask.repeat(1,1,self.args.n_agents).view(-1, 1)
        masked_td_error = td_error * td_mask

        td_loss = (masked_td_error ** 2).sum() / td_mask.sum()

        # Opt loss
        # Computing the targets
        opt_max_actions = th.zeros(size=(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, self.args.n_actions), device=batch.device)
        opt_max_actions_onehot = opt_max_actions.scatter(3, max_actions_current, 1)
        opt_max_actions_onehot_repeat = opt_max_actions_onehot.repeat(1,1,self.args.n_agents,1)
        agent_mask = (1 - th.eye(self.args.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.args.n_actions)
        opt_masked_actions = opt_max_actions_onehot_repeat * agent_mask.unsqueeze(0).unsqueeze(0)
        opt_masked_actions = opt_masked_actions.view(-1, self.args.n_agents * self.args.n_actions)

        opt_target_qs, opt_vs = self.mixer(batch[:,:-1], opt_masked_actions)

        opt_error = max_actions_qvals.squeeze(3).sum(dim=2, keepdim=True).repeat(1,1,self.args.n_agents).view(-1, 1) - opt_target_qs.gather(1, max_actions_current.view(-1, 1)).detach() + opt_vs
        opt_loss = ((opt_error * td_mask) ** 2).sum() / td_mask.sum()

        # NOpt loss
        qsums = chosen_action_qvals.clone().unsqueeze(2).repeat(1,1,self.args.n_agents,1).view(-1, self.args.n_agents)
        ids_to_zero = th.tensor([i for i in range(self.args.n_agents)], device=batch.device).repeat(batch.batch_size * (batch.max_seq_length - 1))
        qsums.scatter(1, ids_to_zero.unsqueeze(1), 0)
        nopt_error = mac_out[:, :-1].contiguous().view(-1, self.args.n_actions) + qsums.sum(dim=1, keepdim=True) - counter_qs.detach() + vs
        min_nopt_error = th.min(nopt_error, dim=1, keepdim=True)[0]
        nopt_loss = ((min_nopt_error * td_mask) ** 2).sum() / td_mask.sum()

        loss = td_loss + self.args.opt_loss * opt_loss + self.args.nopt_min_loss * nopt_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (td_targets * td_mask).sum().item()/td_mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
