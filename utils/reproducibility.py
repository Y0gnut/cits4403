import os
import sys
import hashlib
import json
import platform
import random
from typing import Optional, Dict

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def set_seed(seed: int) -> None:
    """Set global seeds for reproducibility.

    - Sets PYTHONHASHSEED
    - Seeds Python's random
    - Seeds NumPy (if available)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def file_sha256(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """Compute SHA256 of a file if it exists."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def env_info() -> Dict[str, str]:
    """Collect minimal environment and versions for reproducibility logs."""
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    # Lazy imports to avoid hard deps
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        info["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import scipy as _sp
        info["scipy"] = _sp.__version__
    except Exception:
        pass
    try:
        import matplotlib as _mpl
        info["matplotlib"] = _mpl.__version__
    except Exception:
        pass
    return info


def save_run_metadata(path: str, data: Dict) -> None:
    """Save run metadata as pretty JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

