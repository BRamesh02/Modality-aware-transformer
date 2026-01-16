from datetime import date, datetime, timezone
import pandas_market_calendars as mcal
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import re
import hashlib

DEFAULT_KEEP_FIELDS: List[str] = [
    "Date",
    "Stock_symbol",
    "Url",
    "Publisher",
    "Author",
    "Article_title",
    "Article",
    "Textrank_summary",
    "Lexrank_summary",
    "Lsa_summary",
    "Luhn_summary",
]

WS_RE = re.compile(r"\s+")


def clean_str(s: Any) -> Optional[str]:
    """
    Clean a value that may be None:
    - cast to str
    - normalize whitespace (newlines, tabs) to single spaces
    - strip
    - return None if empty
    """
    if s is None:
        return None
    out = WS_RE.sub(" ", str(s)).strip()
    return out if out else None


def parse_dt_utc(date_str: Any) -> Optional[datetime]:
    """
    Parse FNSPID dates of the form: "2020-06-05 06:30:54 UTC".
    Returns a timezone-aware UTC datetime, or None if parsing fails.
    """
    if date_str is None:
        return None

    s = clean_str(date_str)
    if s is None:
        return None

    s = s.replace(" UTC", "").strip()

    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def make_projector(keep_fields: Sequence[str] = DEFAULT_KEEP_FIELDS):
    """
    Return a `project(ex)` function that keeps only `keep_fields`.
    Useful for HF datasets: split.map(make_projector(...)).
    """
    keep_fields = list(keep_fields)

    def project(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {k: ex.get(k, None) for k in keep_fields}

    return project


def build_text_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw FNSPID example into a standardized dict.
    We keep all text variants (article + summaries + title).
    """
    title = clean_str(ex.get("Article_title"))
    article = clean_str(ex.get("Article"))

    textrank = clean_str(ex.get("Textrank_summary"))
    lexrank = clean_str(ex.get("Lexrank_summary"))
    lsa = clean_str(ex.get("Lsa_summary"))
    luhn = clean_str(ex.get("Luhn_summary"))

    return {
        "dt_utc": parse_dt_utc(ex.get("Date")),
        "stock_symbol": clean_str(ex.get("Stock_symbol")),
        "url": clean_str(ex.get("Url")),
        "publisher": clean_str(ex.get("Publisher")),
        "author": clean_str(ex.get("Author")),
        "title": title,
        "article": article,
        "textrank_summary": textrank,
        "lexrank_summary": lexrank,
        "lsa_summary": lsa,
        "luhn_summary": luhn,
    }


def is_valid_record(ex: Dict[str, Any], min_text_len: int = 20) -> bool:
    """
    Minimum filter:
    - valid dt_utc
    - valid stock_symbol
    - at least one non-empty text field among (article, summaries, title)
    - minimum length min_text_len
    """
    if ex.get("dt_utc") is None:
        return False
    if ex.get("stock_symbol") is None:
        return False

    text_fields = [
        ex.get("article"),
        ex.get("textrank_summary"),
        ex.get("lexrank_summary"),
        ex.get("lsa_summary"),
        ex.get("luhn_summary"),
        ex.get("title"),
    ]

    texts = [t for t in text_fields if isinstance(t, str) and len(t) >= min_text_len]
    return len(texts) > 0


def make_dedup_key(ex: Dict[str, Any]) -> str:
    """
    Deduplication key:
    - prefer URL if available
    - otherwise hash (stock_symbol, dt_utc, title)
    """
    url = ex.get("url")
    if url:
        return "U:" + url

    sym = ex.get("stock_symbol") or ""
    dt = ex.get("dt_utc")
    dt_s = dt.isoformat() if isinstance(dt, datetime) else ""
    title = ex.get("title") or ""

    base = f"{sym}|{dt_s}|{title}"
    h = hashlib.blake2b(base.encode("utf-8"), digest_size=16).hexdigest()
    return "H:" + h


def dedup_stream(
    iterable: Iterable[Dict[str, Any]],
    max_seen: int = 500_000,
) -> Iterable[Dict[str, Any]]:
    """
    Approximate streaming deduplication with a bounded cache.
    """
    from collections import deque

    seen = set()
    fifo = deque()

    for ex in iterable:
        key = make_dedup_key(ex)
        if key in seen:
            continue
        seen.add(key)
        fifo.append(key)

        if len(fifo) > max_seen:
            old = fifo.popleft()
            seen.discard(old)

        yield ex


_nyse = mcal.get_calendar("NYSE")
_nyse_schedule = _nyse.schedule(start_date="1999-01-01", end_date="2025-12-31")
_TRADING_DAYS = pd.DatetimeIndex(_nyse_schedule.index)


def next_trading_day_nyse(d: date) -> Optional[date]:
    """
    Return the next NYSE trading day strictly after date d.
    """
    ts = pd.Timestamp(d)
    pos = _TRADING_DAYS.searchsorted(ts + pd.Timedelta(days=1), side="left")
    return None if pos >= len(_TRADING_DAYS) else _TRADING_DAYS[pos].date()


def add_effective_date(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign news dated at day t to the next NYSE trading day (t+1).
    """
    dt = ex.get("dt_utc")
    ex["effective_date"] = (
        next_trading_day_nyse(dt.date()).isoformat() if dt is not None else None
    )
    return ex


def is_trading_day_nyse(d: date) -> bool:
    """
    True if date d is a NYSE trading day.
    """
    ts = pd.Timestamp(d)
    pos = _TRADING_DAYS.searchsorted(ts, side="left")
    return pos < len(_TRADING_DAYS) and _TRADING_DAYS[pos].date() == d


def same_or_next_trading_day_nyse(d: date) -> Optional[date]:
    """
    Return d if it's a trading day; otherwise return the next trading day after d.
    """
    if is_trading_day_nyse(d):
        return d
    return next_trading_day_nyse(d)


def add_effective_date_keep_same(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    effective_date = same day if trading day, else next trading day.
    Stored as ISO string YYYY-MM-DD.
    """
    dt = ex.get("dt_utc")
    ex["effective_date"] = (
        same_or_next_trading_day_nyse(dt.date()).isoformat() if dt is not None else None
    )
    return ex


def norm_url(url: Any) -> Optional[str]:
    """
    Lower + strip. Returns None if empty.
    """
    s = clean_str(url)
    return s.lower() if s else None


def norm_title_key(title: Any) -> Optional[str]:
    """
    Normalization for duplicate detection on title (fallback).
    Keeps it simple: lowercase + collapse spaces.
    """
    s = clean_str(title)
    return s.lower() if s else None


def make_df_dedup_key(
    url: Any,
    stock_symbol: Any,
    pub_day: Any,
    title: Any,
) -> str:
    """
    DataFrame-friendly dedup key:
      - if URL exists: key = U:<url_norm>|<sym>|<pub_day>
      - else: key = H:<hash(title_norm|sym|pub_day)>
    pub_day should be date-like or string 'YYYY-MM-DD'.
    """
    u = norm_url(url)
    sym = clean_str(stock_symbol) or ""
    t = norm_title_key(title) or ""

    if isinstance(pub_day, (datetime, pd.Timestamp)):
        d = pub_day.date().isoformat()
    elif isinstance(pub_day, date):
        d = pub_day.isoformat()
    else:
        d = str(pub_day) if pub_day is not None else ""

    if u:
        return f"U:{u}|{sym}|{d}"

    base = f"{t}|{sym}|{d}"
    h = hashlib.blake2b(base.encode("utf-8"), digest_size=16).hexdigest()
    return "H:" + h


def choose_text_title_first(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Choose a single text field for NLP. Here: title-first (robust for FNSPID).
    Returns (text, source).
    """
    title = ex.get("title")
    if title:
        return title, "title"

    for field, name in [
        ("textrank_summary", "textrank"),
        ("lexrank_summary", "lexrank"),
        ("lsa_summary", "lsa"),
        ("luhn_summary", "luhn"),
        ("article", "article"),
    ]:
        v = ex.get(field)
        if v:
            return v, name

    return None, None


def add_chosen_text(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add:
      - text: chosen string for downstream NLP
      - text_source: which field was used
      - text_len: word count (0 if missing)
    """
    text, src = choose_text_title_first(ex)
    ex["text"] = text
    ex["text_source"] = src
    ex["text_len"] = len(text.split()) if isinstance(text, str) else 0
    return ex
