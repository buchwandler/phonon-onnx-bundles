"""Deterministic, reversible names for flat GitHub Release assets."""
from __future__ import annotations

import re
from pathlib import PurePosixPath


_PREFIX = "bundle--"
_SAFE_BYTES = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_."
)
_ESCAPED_BYTE = re.compile(r"~([0-9A-F]{2})")


def _validate_bundle_path(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError(f"unsafe bundle path: {path!r}")
    pure = PurePosixPath(path)
    parts = tuple(path.split("/"))
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe bundle path: {path!r}")
    return parts


def _encode_component(component: str) -> str:
    encoded: list[str] = []
    for byte in component.encode("utf-8"):
        if byte in _SAFE_BYTES:
            encoded.append(chr(byte))
        else:
            encoded.append(f"~{byte:02X}")
    return "".join(encoded)


def flattened_release_asset_name(path: str) -> str:
    """Encode a bundle-relative POSIX path as a collision-free flat asset name.

    Slashes become the ``--`` component separator. Hyphens and tildes are escaped,
    so the separator and escape sequences cannot occur ambiguously in a component.
    """
    parts = _validate_bundle_path(path)
    return _PREFIX + "--".join(_encode_component(part) for part in parts)


def bundle_path_from_release_asset(name: str) -> str:
    """Decode a name returned by :func:`flattened_release_asset_name`."""
    if not isinstance(name, str) or not name.startswith(_PREFIX):
        raise ValueError(f"not a flattened Phonon bundle asset: {name!r}")
    encoded_parts = name[len(_PREFIX) :].split("--")
    if not encoded_parts or any(not part for part in encoded_parts):
        raise ValueError(f"invalid flattened Phonon bundle asset: {name!r}")
    decoded_parts: list[str] = []
    for part in encoded_parts:
        raw = bytearray()
        index = 0
        while index < len(part):
            if part[index] == "~":
                match = _ESCAPED_BYTE.match(part, index)
                if match is None:
                    raise ValueError(f"invalid escape in flattened asset name: {name!r}")
                raw.append(int(match.group(1), 16))
                index = match.end()
            else:
                char = part[index]
                if ord(char) > 127 or ord(char) not in _SAFE_BYTES:
                    raise ValueError(f"invalid character in flattened asset name: {name!r}")
                raw.append(ord(char))
                index += 1
        try:
            decoded_parts.append(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 in flattened asset name: {name!r}") from exc
    path = "/".join(decoded_parts)
    _validate_bundle_path(path)
    if flattened_release_asset_name(path) != name:
        raise ValueError(f"non-canonical flattened Phonon bundle asset: {name!r}")
    return path


__all__ = ["bundle_path_from_release_asset", "flattened_release_asset_name"]
