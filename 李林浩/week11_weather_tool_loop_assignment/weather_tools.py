import httpx

GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

WEATHER_TEXT = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def search_city(city: str) -> str:
    """根据城市名称查询候选地点，返回经纬度和行政区信息。"""
    params = {
        "name": city,
        "count": 5,
        "language": "zh",
        "format": "json",
    }

    try:
        response = httpx.get(GEOCODING_API, params=params, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"城市查询失败：{exc}"

    results = response.json().get("results") or []
    if not results:
        return f"没有找到城市：{city}"

    lines = []
    for index, item in enumerate(results, start=1):
        name = item.get("name", "")
        country = item.get("country", "")
        admin1 = item.get("admin1", "")
        latitude = item.get("latitude")
        longitude = item.get("longitude")
        population = item.get("population")

        location = " ".join(part for part in [country, admin1, name] if part)
        population_text = f"，人口约 {population}" if population else ""
        lines.append(
            f"{index}. {location}：latitude={latitude}, longitude={longitude}{population_text}"
        )

    return "\n".join(lines)


def query_weather(latitude: float, longitude: float) -> str:
    """根据经纬度查询当前天气和未来三天天气。"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,wind_speed_10m,weather_code"
        ),
        "daily": (
            "weather_code,temperature_2m_max,"
            "temperature_2m_min,precipitation_probability_max"
        ),
        "timezone": "auto",
        "forecast_days": 3,
    }

    try:
        response = httpx.get(FORECAST_API, params=params, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"天气查询失败：{exc}"

    data = response.json()
    current = data.get("current", {})
    daily = data.get("daily", {})

    code = current.get("weather_code")
    current_desc = WEATHER_TEXT.get(code, f"天气代码 {code}")

    lines = [
        f"当前位置坐标：{latitude}, {longitude}",
        f"当前天气：{current_desc}",
        f"当前温度：{current.get('temperature_2m')}°C",
        f"体感温度：{current.get('apparent_temperature')}°C",
        f"相对湿度：{current.get('relative_humidity_2m')}%",
        f"风速：{current.get('wind_speed_10m')} km/h",
        "",
        "未来三天：",
    ]

    dates = daily.get("time", [])
    weather_codes = daily.get("weather_code", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    rain_probs = daily.get("precipitation_probability_max", [])

    for i, date in enumerate(dates):
        day_code = weather_codes[i] if i < len(weather_codes) else None
        desc = WEATHER_TEXT.get(day_code, f"天气代码 {day_code}")
        max_temp = max_temps[i] if i < len(max_temps) else "-"
        min_temp = min_temps[i] if i < len(min_temps) else "-"
        rain_prob = rain_probs[i] if i < len(rain_probs) else "-"
        lines.append(
            f"{date}：{desc}，{min_temp}°C ~ {max_temp}°C，最高降水概率 {rain_prob}%"
        )

    return "\n".join(lines)


TOOL_DISPATCH = {
    "search_city": search_city,
    "query_weather": query_weather,
}
