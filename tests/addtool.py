from mcp.server.fastmcp import FastMCP

from pathlib import Path

mcp: FastMCP = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int):
    """add two numbers"""
    return a + b


@mcp.tool()
def cnt_txt_files() -> int:
    """count py files"""
    cwd = Path(__file__).parent
    py_files = list(cwd.glob("*.py"))
    return len(py_files)


@mcp.tool()
def list_txt_files() -> int:
    """count py files"""
    cwd = Path(__file__).parent
    py_files = list(cwd.glob("*.py"))
    return [str(x) for x in py_files]


if __name__ == "__main__":
    mcp.run()
