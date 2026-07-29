"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费
  4. 拆分为两步：get_city_location（城市名→经纬度）+ get_weather_by_location（经纬度→天气）
     便于 Function Call / MCP / CLI 多轮调用演示

使用方式（作为模块）：
  from src.weather_backend import get_city_location, get_weather_by_location

  # 分步调用（多轮工具调用演示）：
  loc_json = get_city_location("宁德")
  print(get_weather_by_location(26.66, 119.55, city_name="宁德"))

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import json

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


# ── 函数 1：城市名 → 经纬度等地理信息 ──────────────────────────────────────

def get_city_location(city: str) -> str:
    """
    根据城市名获取城市经纬度等地理信息。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        JSON 字符串，包含 city_name、country、admin1、latitude、longitude。
        查询失败时返回可读的错误提示字符串（非 JSON），方便 LLM 直接消费。
    """
    with httpx.Client(timeout=10.0) as client:
        # Geocoding — 城市名 → 经纬度
        # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
        # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        location_info = {
            "city_name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),  # 省/州级行政区
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
        }
        return json.dumps(location_info, ensure_ascii=False)


# ── 函数 2：经纬度 → 天气报告 ──────────────────────────────────────────────

def get_weather_by_location(
    latitude: float,
    longitude: float,
    city_name: str = "",
    country: str = "",
    admin1: str = "",
) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        latitude:  纬度
        longitude: 经度
        city_name: 城市名（仅用于输出展示，可选）
        country:   国家（仅用于输出展示，可选）
        admin1:    省/州（仅用于输出展示，可选）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
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
        location_str = f"{country} {admin1} {city_name}".strip()

        lines = [
            f"【{location_str}】天气报告" if location_str else f"【{latitude:.2f}°N, {longitude:.2f}°E】天气报告",
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


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="天气查询后端")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_loc = sub.add_parser("location", help="城市名 → 经纬度")
    p_loc.add_argument("--city", required=True, help="城市中文名")

    p_wx = sub.add_parser("weather", help="经纬度 → 天气")
    p_wx.add_argument("--latitude", type=float, required=True)
    p_wx.add_argument("--longitude", type=float, required=True)
    p_wx.add_argument("--city-name", default="")
    p_wx.add_argument("--country", default="")
    p_wx.add_argument("--admin1", default="")

    args = parser.parse_args()
    if args.cmd == "location":
        print(get_city_location(args.city))
    elif args.cmd == "weather":
        print(get_weather_by_location(
            args.latitude, args.longitude,
            city_name=args.city_name, country=args.country, admin1=args.admin1,
        ))
