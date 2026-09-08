"""Модуль для завантаження економічного календаря через RSS (наприклад, ForexFactory).

Дозволяє блокувати торгівлю (або переходити в risk-off режим) під час виходу важливих макроекономічних новин.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Приклад RSS-календаря, який можна використовувати
# Багато трейдерів використовують Investing.com або ForexFactory.
# ForexFactory дає зручний XML.
DEFAULT_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"


def fetch_economic_calendar(url: str = DEFAULT_CALENDAR_URL) -> pd.DataFrame:
    """Завантажити економічний календар та повернути DataFrame подій."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Помилка завантаження економічного календаря: %s", exc)
        return pd.DataFrame()

    try:
        root = ET.fromstring(resp.content)
    except Exception as exc:
        logger.error("Помилка парсингу XML економічного календаря: %s", exc)
        return pd.DataFrame()

    events = []
    for event in root.findall("event"):
        title_el = event.find("title")
        country_el = event.find("country")
        date_el = event.find("date")
        time_el = event.find("time")
        impact_el = event.find("impact")
        forecast_el = event.find("forecast")
        previous_el = event.find("previous")

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        country = country_el.text.strip() if country_el is not None and country_el.text else ""
        date_str = date_el.text.strip() if date_el is not None and date_el.text else ""
        time_str = time_el.text.strip() if time_el is not None and time_el.text else ""
        impact = impact_el.text.strip() if impact_el is not None and impact_el.text else ""

        forecast = forecast_el.text.strip() if forecast_el is not None and forecast_el.text else ""
        previous = previous_el.text.strip() if previous_el is not None and previous_el.text else ""

        if not date_str or not time_str:
            continue

        # Формат часу зазвичай: 10:00am, 2:30pm або All Day
        if time_str.lower() == "all day" or time_str.lower() == "tentative":
            # Не беремо події без точного часу для HFT
            continue

        # Дата у форматі: 09-08-2026
        try:
            dt_str = f"{date_str} {time_str}"
            dt = pd.to_datetime(dt_str)
            # Припустимо, що час EST, переводимо в UTC
            # ForexFactory timezone by default is EST
            dt = dt.tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
        except Exception:
            continue

        events.append(
            {"ts": dt, "title": title, "country": country, "impact": impact, "forecast": forecast, "previous": previous}
        )

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    df = df.set_index("ts").sort_index()
    logger.info("Завантажено %d економічних подій", len(df))
    return df


def is_news_time(
    current_time: datetime,
    calendar_df: pd.DataFrame,
    impact_level: str = "High",
    window_before_mins: int = 15,
    window_after_mins: int = 15,
    countries: list[str] | None = None,
) -> bool:
    """Перевіряє, чи поточний час підпадає під вікно виходу новин.

    current_time: час у UTC (наївний)
    """
    if calendar_df.empty:
        return False

    if countries is None:
        countries = ["USD"]

    mask = (calendar_df["impact"] == impact_level) & (calendar_df["country"].isin(countries))
    important_events = calendar_df[mask]

    if important_events.empty:
        return False

    # Перевіряємо, чи поточний час в межах [ts - window_before, ts + window_after]
    for ts, _ in important_events.iterrows():
        start_block = ts - pd.Timedelta(minutes=window_before_mins)
        end_block = ts + pd.Timedelta(minutes=window_after_mins)
        if start_block <= current_time <= end_block:
            return True

    return False
