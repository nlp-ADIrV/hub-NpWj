"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 与 rag_server 共存于不同子进程，由 Host 统一管理——展示 MCP"多 Server 聚合"

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src.weather_backend import get_weather as _get_weather  # noqa: E402
from src.weather_backend import geocode as _geocode  # noqa: E402
from src.weather_backend import get_weather_by_coords as _get_weather_by_coords  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报（一步到位）。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather(city)


@mcp.tool()
def geocode(city: str) -> str:
    """
    查询城市经纬度坐标，返回含行政级别的候选列表。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。

    Returns:
        格式化候选列表，含行政级别、人口、坐标。
    """
    return _geocode(city)


@mcp.tool()
def get_weather_by_coords(lat: float, lon: float) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        lat: 纬度（北正南负）
        lon: 经度（东正西负）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather_by_coords(lat, lon)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
