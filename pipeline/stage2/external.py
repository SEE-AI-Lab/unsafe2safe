"""Import boundary for the external InstructPix2Pix checkout.

Unsafe2Safe keeps the base editor outside this repository. Set
``INSTRUCT_PIX2PIX_ROOT`` to an InstructPix2Pix checkout before importing
``ldm`` modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_external(explicit_root=None) -> Path:
    """Add the external checkout's stable_diffusion directory to sys.path."""
    root = Path(explicit_root or os.environ["INSTRUCT_PIX2PIX_ROOT"]).expanduser().resolve()
    sys.path.insert(0, str(root / "stable_diffusion"))
    return root
