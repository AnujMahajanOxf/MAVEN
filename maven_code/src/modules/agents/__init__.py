REGISTRY = {}

from .rnn_agent import RNNAgent
from .noise_rnn_agent import RNNAgent as NoiseRNNAgent
from .ff_agent import FFAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["noise_rnn"] = NoiseRNNAgent
