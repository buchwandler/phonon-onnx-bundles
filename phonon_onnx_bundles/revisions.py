from __future__ import annotations

import re

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def is_commit_sha(value: object) -> bool:
    return isinstance(value, str) and _COMMIT_SHA_RE.fullmatch(value) is not None
