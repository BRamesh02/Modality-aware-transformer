from datetime import date, datetime, timezone
import pandas_market_calendars as mcal
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Sequence

import re
import hashlib


# Constantes

DEFAULT_KEEP_FIELDS: List[str] = [
    "Date", "Stock_symbol", "Url", "Publisher", "Author",
    "Article_title", "Article",
    "Textrank_summary", "Lexrank_summary", "Lsa_summary", "Luhn_summary",
]

WS_RE = re.compile(r"\s+")


# Helpers de nettoyage

def clean_str(s: Any) -> Optional[str]:
    """
    Nettoie une valeur potentiellement None :
    - convertit en str
    - normalise les espaces (incluant retours ligne, tabs) en espace simple
    - strip
    - renvoie None si vide
    """
    if s is None:
        return None
    out = WS_RE.sub(" ", str(s)).strip()
    return out if out else None


def parse_dt_utc(date_str: Any) -> Optional[datetime]:
    """
    Parse les dates FNSPID de type: "2020-06-05 06:30:54 UTC"
    Retourne un datetime timezone-aware en UTC, ou None si parsing impossible.
    """
    if date_str is None:
        return None

    s = clean_str(date_str)
    if s is None:
        return None

    # Enlever suffixe explicite "UTC" si présent
    s = s.replace(" UTC", "").strip()

    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# Sélection de colonnes


def make_projector(keep_fields: Sequence[str] = DEFAULT_KEEP_FIELDS):
    """
    Renvoie une fonction `project(ex)` qui garde uniquement `keep_fields`.
    Utile pour HF datasets: split.map(make_projector(...))
    """
    keep_fields = list(keep_fields)

    def project(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {k: ex.get(k, None) for k in keep_fields}

    return project


def build_text_record(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforme un exemple FNSPID brut en un dict standardisé.
    Ici, on garde toutes les variantes de texte (article + résumés + titre).
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
        # "text_fallback": article or textrank or lexrank or lsa or luhn or title,
    }


# Validations

def is_valid_record(ex: Dict[str, Any], min_text_len: int = 20) -> bool:
    """
    Filtre minimum :
    - dt_utc OK
    - stock_symbol OK
    - au moins un champ texte non vide parmi (article, résumés, title)
    - longueur minimale min_text_len
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

# Dédoublonnage streaming-friendly (but retirer les quasi doublons)


def make_dedup_key(ex: Dict[str, Any]) -> str:
    """
    Clef de dédoublonnage :
    - priorité à l'URL si disponible,
    - sinon hash de (stock_symbol, dt_utc, title).
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
    Dédoublonnage approximatif en streaming via un cache borné.
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

# Aligner sur business days
# Calendrier NYSE


_nyse = mcal.get_calendar("NYSE")
_nyse_schedule = _nyse.schedule(
    start_date="1999-01-01",
    end_date="2025-12-31"
)
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
        next_trading_day_nyse(dt.date()).isoformat()
        if dt is not None else None
    )
    return ex
