"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 与 rag_server 共存于不同子进程，由 Host 统一管理——展示 MCP"多 Server 聚合"
  3. 天气查询拆分为两个工具：get_city_location（城市名→经纬度）+
     get_weather_by_location（经纬度→天气），便于多轮工具调用演示

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
from src.weather_backend import (  # noqa: E402
    get_city_location as _get_city_location,
    get_weather_by_location as _get_weather_by_location,
)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def get_city_location(city: str) -> str:
    """
    根据城市名获取城市经纬度等地理信息。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        JSON 字符串，包含 city_name、country、admin1、latitude、longitude。
    """
    return _get_city_location(city)


@mcp.tool()
def get_weather_by_location(
    latitude: float,
    longitude: float,
    city_name: str = "",
    country: str = "",
    admin1: str = "",
) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。需先调用 get_city_location 获取经纬度后再调用本工具。

    Args:
        latitude:  纬度，如 26.66
        longitude: 经度，如 119.55
        city_name: 城市名（仅用于展示，可选）
        country:   国家（仅用于展示，可选）
        admin1:    省/州（仅用于展示，可选）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather_by_location(
        latitude, longitude,
        city_name=city_name, country=country, admin1=admin1,
    )


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
