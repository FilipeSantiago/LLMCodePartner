"""Gateway attempt order and bounded operational retry policy."""


def operational_failure(message: str) -> bool:
    """Whether another agent can reasonably recover from this failure."""
    value = message.lower()
    return any(marker in value for marker in (
        "quota", "rate limit", "rate_limit", "usage limit", "session limit",
        "unavailable", "not authenticated", "login required", "unauthorized",
        "authentication", "permission denied", "model_not_found", "model not available",
        "executable not found", "connection", "timed out", "timeout",
        "http 401", "http 403", "http 429", "http 500", "http 502",
        "http 503", "http 504", "status 401", "status 403", "status 429",
        "status 500", "status 502", "status 503", "status 504",
        "bad gateway", "tool call could not be parsed", "tool error",
        "tool failed", "provider error", "stopped without a terminal result",
        "routing candidates failed", "invalid verdict", "invalid json", "malformed", "schema",
    ))


def brief_failure(message: str) -> str:
    """A user-facing failure reason without raw CLI or gateway output."""
    value = message.lower()
    for markers, reason in (
        (("quota", "rate limit", "rate_limit", "usage limit", "session limit", "429"),
         "agent usage limit reached"),
        (("not authenticated", "login required", "unauthorized", "authentication", "401"),
         "agent authentication failed"),
        (("permission denied", "403"), "agent access denied"),
        (("tool call could not be parsed",), "tool call could not be parsed"),
        (("invalid json", "invalid verdict", "malformed", "schema"),
         "agent response was malformed"),
        (("timed out", "timeout"), "agent timed out"),
        (("connection", "bad gateway", "502", "503", "504", "unavailable"),
         "agent connection failed"),
        (("tool error", "tool failed"), "IDE tool failed"),
        (("stopped without a terminal result",), "agent stopped unexpectedly"),
    ):
        if any(marker in value for marker in markers):
            return reason
    return "agent execution failed"


def gateway_attempts(mode: str) -> tuple[bool, ...]:
    if mode == "required":
        return (True,)
    if mode == "disabled":
        return (False,)
    return (True, False)


def same_credential_limit(message: str) -> bool:
    """A direct CLI attempt cannot evade a subscription-wide quota denial."""
    value = message.lower()
    return any(marker in value for marker in (
        "session limit", "usage limit", "quota exceeded", "insufficient_quota",
    ))


def gateway_kwargs(provider: str, gateway_url: str, use_gateway: bool) -> dict:
    if not use_gateway:
        return {}
    if provider == "claude":
        return {"execution_env": {"ANTHROPIC_BASE_URL": gateway_url}}
    return {"gateway_url": gateway_url}
