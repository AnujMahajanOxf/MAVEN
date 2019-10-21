REGISTRY = {}

from .basic_controller import BasicMAC
from .noise_controller import NoiseMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["noise_mac"] = NoiseMAC
