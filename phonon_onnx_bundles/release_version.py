from __future__ import annotations

import re

_RELEASE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")


def validate_release_version(value: str) -> str:
    """Validate the repository's release-set version input format."""
    if not isinstance(value, str) or not _RELEASE_VERSION_RE.fullmatch(value):
        raise ValueError("release_version must be SemVer-like (for example, 0.2.0 or 1.0.0-rc.1)")
    return value
