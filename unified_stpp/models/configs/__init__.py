"""Model-family construction configs.

Each config class owns the parameters and build logic for one model family.
Use BaseModelConfig.from_dict(merged_dict, ...) to instantiate, then
call cfg.build_model() to get a fully wired UnifiedSTPP.
"""

from .base import BaseModelConfig, ConfigRegistry
from .auto_stpp import AutoSTPPConfig
from .auto_stpp_faithful import AutoSTPPFaithfulConfig
from .deep_stpp import DeepSTPPConfig
from .neural_stpp import (
    NeuralSTPPConfig,
    NeuralSTPPJumpSCConfig,
    NeuralSTPPAttnSCConfig,
    NeuralSTPPSharedCondGMMConfig,
    NeuralSTPPSharedJumpCNFConfig,
    NeuralSTPPSharedAttnCNFConfig,
)
from .smash import SMASHConfig
from .diffusion_stpp import DiffusionSTPPConfig
from .factorized import (
    FactorizedConfig, PoissonGMMConfig, HawkesGMMConfig, SelfCorrectingGMMConfig,
    FactorizedCNFConfig, PoissonCNFConfig, HawkesCNFConfig, SelfCorrectingCNFConfig,
    PoissonTVCNFConfig, HawkesTVCNFConfig, SelfCorrectingTVCNFConfig,
)

__all__ = [
    "BaseModelConfig",
    "ConfigRegistry",
    "AutoSTPPConfig",
    "AutoSTPPFaithfulConfig",
    "DeepSTPPConfig",
    "NeuralSTPPConfig",
    "NeuralSTPPJumpSCConfig",
    "NeuralSTPPAttnSCConfig",
    "NeuralSTPPSharedCondGMMConfig",
    "NeuralSTPPSharedJumpCNFConfig",
    "NeuralSTPPSharedAttnCNFConfig",
    "SMASHConfig",
    "DiffusionSTPPConfig",
    "FactorizedConfig",
    "PoissonGMMConfig",
    "HawkesGMMConfig",
    "SelfCorrectingGMMConfig",
    "FactorizedCNFConfig",
    "PoissonCNFConfig",
    "HawkesCNFConfig",
    "SelfCorrectingCNFConfig",
    "PoissonTVCNFConfig",
    "HawkesTVCNFConfig",
    "SelfCorrectingTVCNFConfig",
]
