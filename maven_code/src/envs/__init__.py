from functools import partial
from .multiagentenv import MultiAgentEnv
from .matrix_game.nstep_matrix_game import NStepMatrixGame
from .starcraft2 import StarCraft2Env

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["nstep_matrix"] = partial(env_fn, env=NStepMatrixGame)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
