"""Resolve release scopes before candidate generation."""

import re


VERSION_MENTION = re.compile(
    r"(?<![A-Za-z0-9_.])(?:v|Pydantic[ \t]+v?)([0-9]+(?:\.[0-9]+)*)"
    r"(?![A-Za-z0-9_]|\.(?=[A-Za-z0-9_]))",
    re.IGNORECASE,
)


class SearchError(ValueError):
    def __init__(self, code: str, message: str, status: int = 422, **details) -> None:
        super().__init__(message)
        self.status = status
        self.detail = {"code": code, "message": message, **details}
        self.code = code


def resolve_scope(
    query: str, mode: str, supplied: tuple[str, ...], lock: dict,
) -> tuple[tuple[str, ...], str]:
    releases = lock["releases"]
    options = {"supported_versions": list(releases)}
    aliases = lock["aliases"]

    def canonical(value: str) -> str:
        tag = value.strip().lower()
        tag = aliases.get(tag, tag)
        if tag not in releases:
            raise SearchError("unknown_version", f"Unsupported version: {value}", **options)
        return tag

    mentioned = tuple(dict.fromkeys(
        canonical(f"v{match[1]}") for match in VERSION_MENTION.finditer(query)
    ))
    requested = tuple(dict.fromkeys(canonical(value) for value in supplied))
    if requested and any(tag not in requested for tag in mentioned):
        raise SearchError("version_conflict", "Query mentions a release outside the request scope", **options)

    selected = requested or mentioned
    if not selected and mode == "single":
        return (lock["default_release"],), "default"
    expected = 1 if mode == "single" else 2
    if len(selected) != expected:
        raise SearchError("version_count", f"{mode} requires {expected} distinct release(s)", **options)
    return selected, "request" if requested else "query"
