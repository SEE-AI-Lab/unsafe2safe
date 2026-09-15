"""Import boundary for the external InstructPix2Pix checkout.

Unsafe2Safe keeps the base editor outside this repository. Set
``INSTRUCT_PIX2PIX_ROOT`` to a checkout of InstructPix2Pix, or pass the root to
``configure_external`` before importing ``ldm`` modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _candidate_roots(explicit_root: Optional[str] = None):
    if explicit_root:
        yield Path(explicit_root).expanduser().resolve()
    configured = os.environ.get("INSTRUCT_PIX2PIX_ROOT")
    if configured:
        yield Path(configured).expanduser().resolve()

    # Also support a checkout placed beside the repository or named directly
    # by its ``stable_diffusion/`` subdirectory.
    local_root = Path.cwd()
    yield local_root
    yield local_root / "instruct-pix2pix"


def configure_external(explicit_root: Optional[str] = None) -> Path:
    """Add the external checkout's import roots and return its repository root.

    The adapter accepts either the repository root or its ``stable_diffusion``
    subdirectory so it works with both the upstream checkout and alternate
    local layouts. No files in the checkout are edited.
    """

    seen = set()
    for root in _candidate_roots(explicit_root):
        if root in seen:
            continue
        seen.add(root)
        stable_diffusion = root / "stable_diffusion"
        if (stable_diffusion / "ldm").is_dir():
            import_roots = (stable_diffusion, root)
        elif (root / "ldm").is_dir():
            import_roots = (root,)
        else:
            continue
        for import_root in reversed(import_roots):
            import_root = str(import_root)
            if import_root not in sys.path:
                sys.path.insert(0, import_root)
        return root

    raise ImportError(
        "Could not find InstructPix2Pix. Set INSTRUCT_PIX2PIX_ROOT to its "
        "checkout, for example /path/to/instruct-pix2pix."
    )
