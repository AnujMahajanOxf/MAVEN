from .q_learner import QLearner
from .noise_q_learner import QLearner as NoiseQLearner
from .coma_learner import COMALearner
from .actor_critic_learner import ActorCriticLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["noise_q_learner"] = NoiseQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner