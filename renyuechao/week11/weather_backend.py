"""天气查询后端：城市名 -> 经纬度 -> 当前天气与未来 3 天预报。"""

import httpx


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
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


def get_weather(city: str) -> str:
    """查询指定城市的当前天气和未来 3 天预报。"""
    try:
        with httpx.Client(timeout=10.0) as client:
            geocoding_response = client.get(
                GEOCODING_URL,
                params={
                    "name": city,
                    "count": 10,
                    "language": "zh",
                    "format": "json",
                },
            )
            geocoding_response.raise_for_status()
            locations = geocoding_response.json().get("results") or []

            # 裸地名可能只命中同名村镇；这时用“城市名 + 市”重试一次。
            only_low_level_places = (
                all(
                    str(item.get("feature_code", "")).startswith("PPL")
                    and not str(item.get("feature_code", "")).startswith("PPLA")
                    for item in locations
                )
                if locations
                else True
            )
            has_suffix = any(city.endswith(suffix) for suffix in ("市", "县", "区", "镇"))
            if only_low_level_places and not has_suffix:
                retry_response = client.get(
                    GEOCODING_URL,
                    params={
                        "name": f"{city}市",
                        "count": 10,
                        "language": "zh",
                        "format": "json",
                    },
                )
                retry_response.raise_for_status()
                retry_locations = retry_response.json().get("results") or []
                if retry_locations:
                    locations = retry_locations

            if not locations:
                return f"未找到城市 '{city}'，请换一种写法"

            # 同名地点优先选择行政级别更高、人口更多的结果。
            def rank_location(location: dict) -> tuple[int, int]:
                feature_code = str(location.get("feature_code", ""))
                administrative = int(
                    feature_code.startswith("PPLA") or feature_code.startswith("ADM")
                )
                population = location.get("population") or 0
                return administrative, population

            location = max(locations, key=rank_location)
            latitude = location["latitude"]
            longitude = location["longitude"]

            weather_response = client.get(
                WEATHER_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "wind_speed_10m,weather_code"
                    ),
                    "daily": (
                        "temperature_2m_max,temperature_2m_min,"
                        "precipitation_sum,weather_code"
                    ),
                    "timezone": "Asia/Shanghai",
                    "forecast_days": 3,
                },
            )
            weather_response.raise_for_status()
    except httpx.HTTPError as exc:
        return f"天气数据获取失败：{exc}"

    data = weather_response.json()
    current = data["current"]
    daily = data["daily"]
    current_description = WEATHER_CODE_MAP.get(
        current["weather_code"], f"代码 {current['weather_code']}"
    )
    location_name = " ".join(
        filter(
            None,
            [
                location.get("country", ""),
                location.get("admin1", ""),
                location.get("name", city),
            ],
        )
    )

    lines = [
        f"【{location_name}】天气报告",
        f"当前天气：{current_description}",
        f"温度：{current['temperature_2m']}°C",
        f"相对湿度：{current['relative_humidity_2m']}%",
        f"风速：{current['wind_speed_10m']} km/h",
        "未来 3 天：",
    ]

    for index in range(3):
        description = WEATHER_CODE_MAP.get(daily["weather_code"][index], "未知")
        lines.append(
            f"{daily['time'][index]}：{description}，"
            f"{daily['temperature_2m_max'][index]}°C / "
            f"{daily['temperature_2m_min'][index]}°C，"
            f"降水 {daily['precipitation_sum'][index]} mm"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="直接查询天气后端")
    parser.add_argument("--city", required=True, help="城市中文名，例如：北京")
    arguments = parser.parse_args()
    print(get_weather(arguments.city))
