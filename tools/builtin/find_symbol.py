from pydantic import BaseModel, Field

from context.code_index import build_workspace_code_index
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class FindSymbolParams(BaseModel):
    query: str = Field(
        ...,
        description="Symbol name or fragment to search for. Use '*' to list notable symbols.",
    )
    kind: str | None = Field(
        None,
        description="Optional symbol kind filter such as function, class, struct, enum, interface, type, trait, or macro.",
    )
    language: str | None = Field(
        None,
        description="Optional language filter such as python, c, c++, javascript, typescript, java, go, or rust.",
    )
    limit: int = Field(
        8,
        ge=1,
        le=25,
        description="Maximum number of matching symbols to return.",
    )


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = (
        "Search a lightweight codebase index for functions, classes, structs, enums, interfaces, "
        "types, traits, and macros across common source files."
    )
    kind = ToolKind.READ
    schema = FindSymbolParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = FindSymbolParams(**invocation.params)
        index = build_workspace_code_index(invocation.cwd)

        if not index or index.indexed_files == 0:
            return ToolResult.success_result(
                "No indexable source files were found in the current workspace.",
                metadata={"matches": 0, "query": params.query},
            )

        matches = index.find(
            params.query,
            kind=params.kind,
            language=params.language,
            limit=params.limit,
        )

        if not matches:
            return ToolResult.success_result(
                f"No symbols matched '{params.query}'.",
                metadata={
                    "matches": 0,
                    "query": params.query,
                    "indexed_files": index.indexed_files,
                    "indexed_symbols": len(index.symbols),
                },
            )

        lines = [f"Matches for '{params.query}' ({len(matches)}):"]
        for symbol in matches:
            lines.append(
                f"- {symbol.kind} {symbol.name} [{symbol.language}]"
                f" - {symbol.display_location(index.root)}"
            )
            if symbol.signature:
                lines.append(f"  signature: {symbol.signature}")

        return ToolResult.success_result(
            "\n".join(lines),
            metadata={
                "matches": len(matches),
                "query": params.query,
                "indexed_files": index.indexed_files,
                "indexed_symbols": len(index.symbols),
            },
        )
