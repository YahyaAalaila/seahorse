try:
    from .lightning_module import STPPLightningModule
    from .data_module import STPPDataModule
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
