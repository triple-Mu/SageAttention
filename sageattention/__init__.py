# The import order in this file is functional (torch -> _C -> ops -> core), so
# it is exempt from import sorting: merging or reordering these statements
# would break extension loading and operator registration.
# isort: skip_file

import torch  # noqa: F401  (must be imported before the extension)

# Loading _C runs the TORCH_LIBRARY static initializers and registers every
# torch.ops.sageattention.* operator. Nothing else is exported from _C itself
# (it only carries build-info string constants).
from . import _C  # noqa: F401
from . import ops  # noqa: F401  (register_fake; requires _C first)

from .core import sageattn
from .varlen import sageattn_varlen

__all__ = ["sageattn", "sageattn_varlen"]
__version__ = _C.version
