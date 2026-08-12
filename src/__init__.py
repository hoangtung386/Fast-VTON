"""SwiftEdit: lightning-fast text-guided image editing via one-step diffusion.

The package is organised as:

* :mod:`src.attention` - attention processors and the ARaM controller
* :mod:`src.models` - inversion network, generator, frozen dependencies
* :mod:`src.pipelines` - end-to-end editing entry points
* :mod:`src.vton` - the virtual try-on adaptation (see ``docs/VTON_PLAN.md``)
"""

from src.attention import MaskController
from src.models import AuxiliaryModel, InverseModel, IPSBV2Model
from src.pipelines import EditConfig, edit_image

__version__ = "0.1.0"

__all__ = [
    "AuxiliaryModel",
    "EditConfig",
    "IPSBV2Model",
    "InverseModel",
    "MaskController",
    "__version__",
    "edit_image",
]
