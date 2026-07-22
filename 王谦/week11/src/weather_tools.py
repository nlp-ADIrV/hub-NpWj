"""
weather_tools.py — 作业：把原 get_weather 拆成两个独立工具

  原始 src/weather_backend.get_weather 内部串了两次 HTTP 请求：
    1) Geocoding：城市名 → 经纬度
    2) Forecast：经纬度 → 天气
  本文件把它们拆成两个对 LLM 暴露的独立工具：
    - getcode(city)            ：只做"城市名 → 经纬度"，能独立回答"北京的经纬度"
    - get_weather_by_coords(lat, lon)：只做"经纬度 → 天气"，用户给经纬度即可直接答
  两个工具互不依赖，但模型可以把它们链起来：geocode → 拿经纬度 → get_weather_by_coords，
  这正是 agent loop 的价值——模型自己决定调几次、调哪个，宿主只负责按调用循环执行。

依赖：pip install httpx   （Open-Meteo 免费、无需 key）
"""

import httpx
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

def getcode(city):
    """城市名 → 经纬度"""
    with httpx.Client(timeout=10) as client:
        resp = client.get(GEOCODING_URL, params={
            "name": city, "count": 10, "language": "zh", "format": "json",
        })
        resp.raise_for_status()
        results = resp.json().get("results") or []

        # 与原 backend 同样的同名小村庄消歧策略：裸低级行政点且没带"市/县/区"后缀，
        # 就用 city+"市" 重查一次并优先采用。
        def _getcode(name: str):
            r = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            r.raise_for_status()
            return r.json().get("results") or []
        #Open-Meteo 的 feature_code 里，PPL 开头是普通居民点（村庄 / 乡镇），PPLA 开头是一级行政中心（市、省会）。
        #all(...) 表示：所有结果都满足「是 PPL 开头、且不是 PPLA 开头」，才返回 True。
        #if results else True：如果本来就没搜到结果，也视为「低级行政点」，走重试逻辑。
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        #判断输入城市名是否带行政后缀
        has_suffix = any(city.endswith(suffix) for suffix in ("市", "县", "区","镇"))

        if is_low_admin and not has_suffix:
            rs = _getcode(city+"市")
            if rs:
                results = rs

        if not results:
            return f"未找到城市 {city} 的经纬度。" #如果没找到结果，返回提示信息
        #打分排序 按照行政级别和人口数量排序
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            return (admin_priority, r.get("population", 0))
        #用 max 配合 _rank 函数，从所有结果里选出优先级最高的那个地点。
        loc = max(results, key=_rank)

        lat = loc.get("latitude")
        lon = loc.get("longitude")
        location_str = f"{loc.get('country')}, {loc.get('admin1')}, {loc.get('name')}"

        return(
            f"城市：{location_str}\n"
            f"经度：{lon}\n"
            f"纬度：{lat}"
        )
    
def get_weather_by_coords(lat:float, lon:float):
    """经纬度 → 天气"""
    with httpx.Client(timeout=10) as client:
        try:
            resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"
        
        data = resp.json()
        cur = data.get("current")
        daily = data.get("daily")
        weather_desc = WEATHER_CODE_MAP.get(cur.get("weather_code"), "未知")

        #用列表拼接返回文本，先写坐标、当前天气的各项详情。
        lines = [
            f"经度：{lon}",
            f"纬度：{lat}",
            f"当前温度：{cur.get('temperature_2m')}°C",
            f"相对湿度：{cur.get('relative_humidity_2m')}%",
            f"风速：{cur.get('wind_speed_10m')} m/s",
            f"天气状况：{weather_desc}",
            "",
            "未来三天天气预报："
        ]
        #再写未来三天的天气预报，按天逐行拼接。
        for i in range(3):
            day_w = daily["weather_code"][i]
            day_weather_desc = WEATHER_CODE_MAP.get(day_w, "未知")
            lines.append(
                f"  {daily['time'][i]}：{day_weather_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )
        return "\n".join(lines)

if __name__ == "__main__":
    # 自测：getcode → 拿经纬度 → get_weather_by_coords，手动演示一遍链式调用
    info = getcode("北京")
    print(info)
    # 从文本里把经纬度抠出来继续查天气（仅自测用，模型链式调用时自己解析）
    import re
    m_lat = re.search(r"纬度.*?：(-?\d+\.?\d*)", info)
    m_lon = re.search(r"经度.*?：(-?\d+\.?\d*)", info)
    if m_lat and m_lon:
        print("\n--- 链式调用：拿上面经纬度查天气 ---")
        print(get_weather_by_coords(float(m_lat.group(1)), float(m_lon.group(1))))