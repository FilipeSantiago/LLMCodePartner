"""Gateway attempt order shared by read-only planner and reviewer calls."""


def gateway_attempts(mode: str) -> tuple[bool, ...]:
    if mode == "required":
        return (True,)
    if mode == "disabled":
        return (False,)
    return (True, False)


def gateway_kwargs(provider: str, gateway_url: str, use_gateway: bool) -> dict:
    if not use_gateway:
        return {}
    if provider == "claude":
        return {"execution_env": {"ANTHROPIC_BASE_URL": gateway_url}}
    return {"gateway_url": gateway_url}
