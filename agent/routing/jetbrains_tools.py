"""Implementation-only proxy for PyCharm's standalone MCP server."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent.spec.jetbrains_mcp import JetBrainsMcpClient, JetBrainsMcpError
from mcp_bridge.registry import ToolSpec, is_mutating_tool

DirectToolHandler = Callable[[dict], Awaitable[str]]


class JetBrainsWriteCapabilityError(JetBrainsMcpError):
    """PyCharm is reachable but exposes no supported source mutation tool."""


@dataclass(frozen=True)
class DirectIdeTools:
    """Discovered standalone MCP write tools plus their backend handlers."""

    specs: tuple[ToolSpec, ...]
    handlers: dict[str, DirectToolHandler]

    @classmethod
    async def discover(cls, client: JetBrainsMcpClient | None = None) -> "DirectIdeTools":
        client = client or JetBrainsMcpClient()
        tools = await client.list_tools()
        selected = [tool for tool in tools if is_mutating_tool(tool.name)]
        if not selected:
            available = ", ".join(tool.name for tool in tools) or "none"
            raise JetBrainsWriteCapabilityError(
                "PyCharm standalone MCP server exposes no supported source-file write tool "
                f"(available: {available})"
            )

        async def call(name: str, arguments: dict) -> str:
            return await client.call_tool(name, arguments)

        return cls(
            specs=tuple(ToolSpec(tool.name, tool.description, tool.schema) for tool in selected),
            handlers={tool.name: (lambda arguments, name=tool.name: call(name, arguments))
                      for tool in selected},
        )

    def merge(self, advertised: list[ToolSpec]) -> list[ToolSpec]:
        """Direct IDE writes take precedence over same-named chat tools."""
        direct_names = set(self.handlers)
        return [tool for tool in advertised if tool.name not in direct_names] + list(self.specs)
