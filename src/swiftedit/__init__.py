"""SwiftEdit: lightning-fast text-guided image editing via one-step diffusion.

The package is organised as:

* :mod:`swiftedit.attention` - attention processors and the ARaM controller
* :mod:`swiftedit.models` - inversion network, generator, frozen dependencies
* :mod:`swiftedit.pipelines` - end-to-end editing entry points
* :mod:`swiftedit.vton` - the virtual try-on adaptation (see ``docs/VTON_PLAN.md``)
"""

from swiftedit.attention import MaskController
from swiftedit.models import AuxiliaryModel, InverseModel, IPSBV2Model
from swiftedit.pipelines import EditConfig, edit_image

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
