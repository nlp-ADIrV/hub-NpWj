"""
weather_backend.py — 天气查询后端（拆成两步，供循环工具调用使用）

改造要点：
  原项目 get_weather(city) 在一次函数内完成「地理编码 + 天气查询」。
  作业要求演示「循环调用 / 链式依赖」，因此拆成两个独立工具：
    1. geocode_city(city)           → 城市名 → 经纬度
    2. get_weather_by_coords(...)   → 经纬度 → 天气报告

  Agent 必须先调第 1 个工具，拿到坐标后再调第 2 个——单轮闭环做不到，
  必须用多轮 while 循环（见 run_weather_loop.py）。

依赖：
  pip install httpx
  Open-Meteo API 免费，无需注册
"""

from __future__ import annotations

import json
from typing import Any

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _geocode(client: httpx.Client, name: str) -> list[dict[str, Any]]:
    resp = client.get(GEOCODING_URL, params={
        "name": name, "count": 10, "language": "zh", "format": "json",
    })
    resp.raise_for_status()
    return resp.json().get("results") or []


def _rank(r: dict[str, Any]):
    """行政级别更高、人口更多者优先。"""
    fc = str(r.get("feature_code", ""))
    admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
    pop = r.get("population") or 0
    return (admin_priority, pop)


def geocode_city(city: str) -> str:
    """
    将城市名解析为经纬度等信息（工具 1）。

    Returns:
        JSON 字符串，含 latitude / longitude / name / country / admin1；
        失败时返回可读错误信息。
    """
    with httpx.Client(timeout=10.0) as client:
        results = _geocode(client, city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(client, city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        loc = max(results, key=_rank)
        payload = {
            "city_query": city,
            "name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "feature_code": loc.get("feature_code", ""),
            "population": loc.get("population"),
        }
        return json.dumps(payload, ensure_ascii=False)


def get_weather_by_coords(
    latitude: float,
    longitude: float,
    city_name: str = "",
    country: str = "",
    admin1: str = "",
) -> str:
    """
    根据经纬度查询当前天气及未来 3 天预报（工具 2）。

    通常在 geocode_city 成功之后调用；参数可从工具 1 的 JSON 结果中取出。
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        location_str = f"{country} {admin1} {city_name}".strip() or f"{latitude:.2f},{longitude:.2f}"

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{latitude:.2f}°N, {longitude:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )
        return "\n".join(lines)


def get_weather(city: str) -> str:
    """兼容原接口：内部串行调用两步（非 LLM 循环时可用）。"""
    geo_raw = geocode_city(city)
    try:
        geo = json.loads(geo_raw)
    except json.JSONDecodeError:
        return geo_raw
    return get_weather_by_coords(
        latitude=geo["latitude"],
        longitude=geo["longitude"],
        city_name=geo.get("name", city),
        country=geo.get("country", ""),
        admin1=geo.get("admin1", ""),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--step", choices=["all", "geocode"], default="all")
    args = parser.parse_args()
    if args.step == "geocode":
        print(geocode_city(args.city))
    else:
        print(get_weather(args.city))
