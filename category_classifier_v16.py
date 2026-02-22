"""
Product Category CLASSIFIER V16 - Gemini 3 Flash Preview (Stable Batching + Top-K Taxonomy + UNKNOWN Second Pass)
"""

from __future__ import annotations

import os
import re
import json
import time
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple, Iterable
from collections import Counter
from datetime import datetime
from functools import lru_cache

import pandas as pd
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

# ============================ LOGGING ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ============================ CONFIGURATION ============================

INPUT_FILE = "tf_menu_ramedan.csv"
OUTPUT_FILE = "tf_menu_labeled_v16_ramedan.csv"
TAXONOMY_FILE = "category_definitions_ramedan.json"

CONFIG: Dict[str, Any] = {
    # --- Gemini ---
    "GEMINI_API_KEY": os.getenv("GENAI_API_KEY", "AIzaSyAtRl795HzWoxSNpfLbJ9cw9bxd9kwAFwk"),
    "MODEL_NAME": "gemini-3-flash-preview",

    # Keep temperature as you were doing (and as you prefer) for reasoning.
    # Consistency comes from batching + TopK + deterministic tie-breaks.
    "TEMPERATURE": 1.0,
    "TOP_P": 0.95,
    "TOP_K": 20,

    # --- Processing ---
    "MAX_WORKERS": 5,
    "RATE_LIMIT_PER_SEC": 4,
    "BATCH_SIZE": 20,

    # --- Candidate shortlist (NO full taxonomy shipping) ---
    "CANDIDATE_TOP_K": 18,
    "CANDIDATE_MIN_KEYS": 12,

    # Second-pass for UNKNOWN only
    "SECOND_PASS_ENABLED": True,
    "CANDIDATE_TOP_K_SECOND": 40,
    "CANDIDATE_MIN_KEYS_SECOND": 28,

    # include valid arrow targets (→) in the candidate set
    "INCLUDE_ARROW_TARGETS": True,

    # --- Retries ---
    "MAX_RETRIES": 3,
    "RETRY_DELAY_SEC": 3.0,

    # --- Concurrency ---
    "MAX_IN_FLIGHT_MULTIPLIER": 1,

    # --- Telegram ---
    "TELEGRAM_ENABLED": True,
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8205938582:AAG-fhOjW4tMPkNRpYU8J_Xg7vgMLisHCBU"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "-5091693030"),
    "TELEGRAM_TIMEOUT_SEC": 15,
    "TELEGRAM_MAX_MSG_CHARS": 4000,

    # --- Reporting ---
    "REPORT_EVERY_SECONDS": 60,

    # --- Prompt truncation ---
    "TRUNC_TITLE_CHARS": 220,
    "TRUNC_DESC_CHARS": 520,
    "TRUNC_CTX_CHARS": 120,

    # --- Watchdog ---
    "WATCHDOG_ENABLED": False,
    "MIN_ITEMS_PER_MINUTE": 0.000000001,

    # --- Output encoding ---
    "OUTPUT_ENCODING": "utf-8-sig",
}

if not CONFIG["GEMINI_API_KEY"]:
    raise RuntimeError("GENAI_API_KEY is not set. Please set env var GENAI_API_KEY and re-run.")

genai.configure(api_key=CONFIG["GEMINI_API_KEY"])

# ============================ GLOBAL SHUTDOWN FLAG ============================

class ShutdownFlag:
    def __init__(self):
        self._should_shutdown = False
        self._reason = ""
        self._lock = threading.Lock()

    def set(self, reason: str):
        with self._lock:
            if not self._should_shutdown:
                self._should_shutdown = True
                self._reason = reason

    def is_set(self) -> bool:
        with self._lock:
            return self._should_shutdown

    def reason(self) -> str:
        with self._lock:
            return self._reason

shutdown_flag = ShutdownFlag()

# ============================ RATE LIMITER ============================

class RateLimiter:
    def __init__(self, max_calls_per_second: float):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter(CONFIG["RATE_LIMIT_PER_SEC"])

# ============================ TELEGRAM REPORTER ============================

class TelegramReporter:
    def __init__(self, enabled: bool, bot_token: str, chat_id: str, timeout_sec: int, max_chars: int):
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_sec = timeout_sec
        self.max_chars = max_chars

    def _chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            nl = text.rfind("\n", start, end)
            if nl != -1 and nl > start + 200:
                end = nl
            part = text[start:end].strip()
            if part:
                chunks.append(part)
            start = end
        return chunks

    def send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload_base = {"chat_id": self.chat_id, "disable_web_page_preview": True}

        for part in self._chunk_text(text):
            payload = dict(payload_base)
            payload["text"] = part
            for attempt in range(3):
                try:
                    r = requests.post(url, json=payload, timeout=self.timeout_sec)
                    if r.status_code == 200:
                        break
                    time.sleep(1.5 * (attempt + 1))
                except Exception:
                    time.sleep(1.5 * (attempt + 1))

telegram = TelegramReporter(
    enabled=CONFIG["TELEGRAM_ENABLED"],
    bot_token=CONFIG["TELEGRAM_BOT_TOKEN"],
    chat_id=CONFIG["TELEGRAM_CHAT_ID"],
    timeout_sec=CONFIG["TELEGRAM_TIMEOUT_SEC"],
    max_chars=CONFIG["TELEGRAM_MAX_MSG_CHARS"],
)

# ============================ COST TRACKING ============================

@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    input_cost_per_1m: float = 0.50
    output_cost_per_1m: float = 3.00
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, in_tokens: int, out_tokens: int):
        with self.lock:
            self.input_tokens += int(in_tokens)
            self.output_tokens += int(out_tokens)
            self.calls += 1

    def summary_str(self) -> str:
        with self.lock:
            in_cost = (self.input_tokens / 1_000_000) * self.input_cost_per_1m
            out_cost = (self.output_tokens / 1_000_000) * self.output_cost_per_1m
            total = in_cost + out_cost
            return (
                f"Calls: {self.calls} | "
                f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
                f"Cost: ${total:.4f} (in ${in_cost:.4f} + out ${out_cost:.4f})"
            )

cost_tracker = CostTracker()

# ============================ STATS TRACKING ============================

@dataclass
class StatsTracker:
    processed: int = 0
    success: int = 0
    unknown: int = 0
    error: int = 0
    second_pass_attempts: int = 0
    second_pass_fixes: int = 0
    level1_counts: Counter = field(default_factory=Counter)
    level12_counts: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_from_rows(self, rows: List[Dict[str, Any]]):
        with self.lock:
            for r in rows:
                self.processed += 1
                l1 = str(r.get("level_1", "ERROR"))
                l2 = str(r.get("level_2", "ERROR"))

                if l1 == "ERROR" or l2 == "ERROR":
                    self.error += 1
                elif l1 == "UNKNOWN" or l2 == "UNKNOWN":
                    self.unknown += 1
                else:
                    self.success += 1

                self.level1_counts[l1] += 1
                self.level12_counts[(l1, l2)] += 1

    def inc_second_pass(self, attempts: int, fixes: int):
        with self.lock:
            self.second_pass_attempts += int(attempts)
            self.second_pass_fixes += int(fixes)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "processed": self.processed,
                "success": self.success,
                "unknown": self.unknown,
                "error": self.error,
                "second_pass_attempts": self.second_pass_attempts,
                "second_pass_fixes": self.second_pass_fixes,
                "level1_counts": self.level1_counts.copy(),
                "level12_counts": self.level12_counts.copy(),
            }

stats = StatsTracker()

# ============================ ERROR NOTIFIER ============================

class ErrorNotifier:
    def __init__(self, telegram_reporter: TelegramReporter):
        self.telegram = telegram_reporter
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.last_notification_time: Dict[str, float] = {}
        self.min_notification_interval = 60

    def notify_error(self, error_type: str, error_msg: str, batch_info: str = ""):
        with self.lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            count = self.error_counts[error_type]

            now = time.time()
            last_time = self.last_notification_time.get(error_type, 0)
            if now - last_time < self.min_notification_interval:
                return

            self.last_notification_time[error_type] = now

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"🚨 ERROR ALERT @ {ts}\n"
                f"Type: {error_type}\n"
                f"Occurrence: #{count}\n"
            )
            if batch_info:
                msg += f"Batch: {batch_info}\n"
            msg += f"Message: {error_msg[:500]}\n"
            msg += f"\nStats: {stats.processed:,} items processed so far"

            self.telegram.send(msg)

error_notifier = ErrorNotifier(telegram)

# ============================ RATE WATCHDOG ============================

class RateWatchdog:
    def __init__(self, min_items_per_minute: float):
        self.min_items_per_minute = min_items_per_minute
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.last_check_processed = 0
        self.lock = threading.Lock()

    def check_rate(self, current_processed: int) -> Tuple[bool, str]:
        with self.lock:
            now = time.time()

            if now - self.last_check_time < 120:
                return (False, "")
            if now - self.start_time < 300:
                return (False, "")

            time_elapsed_min = (now - self.last_check_time) / 60.0
            items_processed = current_processed - self.last_check_processed

            if time_elapsed_min > 0:
                rate = items_processed / time_elapsed_min
                self.last_check_time = now
                self.last_check_processed = current_processed

                if rate < self.min_items_per_minute:
                    reason = (
                        f"Processing rate too slow: {rate:.1f} items/min "
                        f"(minimum: {self.min_items_per_minute} items/min). "
                        f"Likely hitting quota limits."
                    )
                    return (True, reason)

            return (False, "")

rate_watchdog = RateWatchdog(CONFIG["MIN_ITEMS_PER_MINUTE"])

# ============================ HELPERS ============================

def normalize_item_id(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip().replace(",", "")

def _clean_text(x: Any, max_len: int) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).replace('"', "").replace("\n", " ").strip()
    if max_len > 0 and len(s) > max_len:
        s = s[:max_len].rstrip()
    return s

def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * part / total):.2f}%"

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def get_processed_ids(filepath: str) -> Set[str]:
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_csv(filepath, usecols=["item_id"], dtype={"item_id": "string"})
        return set(df["item_id"].astype(str))
    except Exception:
        return set()

def save_final_reports(level1_counts: Counter, level12_counts: Counter):
    pd.DataFrame([{"level_1": k, "count": v} for k, v in level1_counts.most_common()]).to_csv(
        "final_level1_counts.csv", index=False, encoding=CONFIG["OUTPUT_ENCODING"]
    )
    pd.DataFrame([{"level_1": k[0], "level_2": k[1], "count": v} for k, v in level12_counts.most_common()]).to_csv(
        "final_level2_counts.csv", index=False, encoding=CONFIG["OUTPUT_ENCODING"]
    )

def format_progress_message(snap: Dict[str, Any]) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]
    sp_attempts = snap.get("second_pass_attempts", 0)
    sp_fixes = snap.get("second_pass_fixes", 0)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[tf_menu] Minute report @ {ts}\n"
        f"Processed: {processed:,}\n"
        f"✅ Success: {success:,} ({_pct(success, processed)})\n"
        f"❓ Unknown: {unknown:,} ({_pct(unknown, processed)})\n"
        f"❌ Error:   {error:,} ({_pct(error, processed)})\n"
        f"🔁 2nd Pass Attempts: {sp_attempts:,} | Fixes: {sp_fixes:,}\n\n"
        f"[COST]\n{cost_tracker.summary_str()}\n"
    )

def format_shutdown_message(reason: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snap = stats.snapshot()
    return (
        f"[tf_menu] 🚨 SHUTDOWN TRIGGERED @ {ts}\n"
        f"Reason: {reason}\n\n"
        f"Progress so far:\n"
        f"Processed: {snap['processed']:,}\n"
        f"✅ Success: {snap['success']:,}\n"
        f"❓ Unknown: {snap['unknown']:,}\n"
        f"❌ Error:   {snap['error']:,}\n"
        f"🔁 2nd Pass Attempts: {snap.get('second_pass_attempts', 0):,} | Fixes: {snap.get('second_pass_fixes', 0):,}\n\n"
        f"[COST]\n{cost_tracker.summary_str()}\n"
        f"\nThe app has stopped to prevent further API calls."
    )

def format_final_message(snap: Dict[str, Any], top_n_l1: int = 40, top_n_l12: int = 60) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]
    sp_attempts = snap.get("second_pass_attempts", 0)
    sp_fixes = snap.get("second_pass_fixes", 0)

    level1_counts: Counter = snap["level1_counts"]
    level12_counts: Counter = snap["level12_counts"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"[tf_menu] FINAL report @ {ts}")
    lines.append(f"Processed: {processed:,}")
    lines.append(f"✅ Success: {success:,}")
    lines.append(f"❓ Unknown: {unknown:,}")
    lines.append(f"❌ Error:   {error:,}")
    lines.append(f"🔁 2nd Pass Attempts: {sp_attempts:,} | Fixes: {sp_fixes:,}")
    lines.append("")
    lines.append("[COST]")
    lines.append(cost_tracker.summary_str())
    lines.append("")
    lines.append(f"Top Level 1 categories (top {top_n_l1}):")
    for k, v in level1_counts.most_common(top_n_l1):
        lines.append(f"- {k}: {v:,}")
    lines.append("")
    lines.append(f"Top Level 1 → Level 2 pairs (top {top_n_l12}):")
    for (l1, l2), v in level12_counts.most_common(top_n_l12):
        lines.append(f"- {l1} → {l2}: {v:,}")
    lines.append("")
    lines.append("Saved CSVs: final_level1_counts.csv, final_level2_counts.csv")
    return "\n".join(lines)

# ============================ ERROR DETECTION ============================

def is_critical_gemini_error(err: str) -> bool:
    e = (err or "").lower()
    if "status: 403" in e or "received http2 header with status: 403" in e:
        return True
    auth_phrases = ["permission denied", "authentication", "invalid api key", "unauthorized", "forbidden"]
    if any(p in e for p in auth_phrases):
        return True
    return False

def is_transient_gemini_error(err: str) -> bool:
    e = (err or "").lower()
    if "429" in e:
        return True
    if "504" in e:
        return True
    if "deadline expired" in e or "deadline_exceeded" in e:
        return True
    for code in ["500", "502", "503"]:
        if code in e:
            return True
    transient_phrases = [
        "connection reset", "connection aborted", "timed out", "timeout", "tls",
        "socket", "temporarily unavailable", "server closed", "broken pipe",
        "unavailable", "cancelled", "canceled", "stream cancelled", "stream canceled",
    ]
    if any(p in e for p in transient_phrases):
        return True
    return False

# ============================ JSON PARSING (robust) ============================

def parse_json_strict(text: str) -> Any:
    raw = (text or "").strip()
    cleaned = _strip_code_fences(raw)

    # best case
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # salvage bracket/brace block
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")

    candidates: List[str] = []
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw[first_brace:last_brace + 1])
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(raw[first_bracket:last_bracket + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    raise ValueError("Failed to parse JSON from model response")

# ============================ PERSIAN NORMALIZATION ============================

_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ئ": "ی",
    "ۀ": "ه",
    "‌": " ",   # ZWNJ -> space (stabilize sorting)
    "ـ": "",    # tatweel
})

_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_SPACES_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06FF]+")

@lru_cache(maxsize=250_000)
def normalize_title_for_sort(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = t.translate(_ARABIC_TO_PERSIAN)
    t = _DIACRITICS_RE.sub("", t)
    t = t.lower()
    t = _SPACES_RE.sub(" ", t).strip()
    return t

def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    t = normalize_title_for_sort(text)
    return set(_TOKEN_RE.findall(t))

# ============================ TAXONOMY INDEX + ARROW TARGET EXTRACTION ============================

_ARROW_RE = re.compile(r"→\s*([^\n؛\)\.]+)")

def extract_arrow_targets(exclusions: str) -> List[str]:
    if not exclusions:
        return []
    targets: List[str] = []
    for m in _ARROW_RE.finditer(exclusions):
        cand = (m.group(1) or "").strip()
        cand = cand.strip(" \t\r\n\"'،;؛.")
        if cand:
            targets.append(cand)
    return targets

class TaxonomyIndex:
    """
    Lightweight lexical index to select Top-K candidate keys per batch.
    This prevents sending the entire taxonomy JSON in every request.
    """

    def __init__(self, taxonomy: Dict[str, Any]):
        self.taxonomy = taxonomy
        self.all_keys: List[str] = sorted(list(taxonomy.keys()))
        self.valid_level_2_keys: Set[str] = set(self.all_keys)

        # Precompute tokens for key and (explanation+exclusions)
        self.key_tokens: Dict[str, Set[str]] = {}
        self.body_tokens: Dict[str, Set[str]] = {}
        self.arrow_targets: Dict[str, List[str]] = {}

        for k, v in taxonomy.items():
            expl = str(v.get("explanation", "") or "")
            excl = str(v.get("exclusions", "") or "")
            body = f"{expl} {excl}"

            self.key_tokens[k] = tokenize(k)
            self.body_tokens[k] = tokenize(body)
            self.arrow_targets[k] = extract_arrow_targets(excl)

        # domain hints by business_line (edit patterns/keys to match your real business_line values)
        self.business_line_hints: List[Tuple[re.Pattern, List[str]]] = [
            (re.compile(r"(cafe|کافه|coffee|قهوه)", re.I), [
                "نوشیدنی گرم بر پایه قهوه",
                "نوشیدنی گرم غیر قهوه",
                "نوشیدنی سرد بر پایه قهوه",
                "شیک",
                "اسموتی و گلاسه",
                "کیک و دسر",
                "کیک روز",
                "کوکی",
                "کروسان",
                "دان و پودر قهوه",
                "گرانولا",
            ]),
            (re.compile(r"(fast|فست|pizza|پیتزا|burger|برگر|ساندویچ|wrap|رپ)", re.I), [
                "پیتزا", "برگر", "ساندویچ", "رپ", "سوخاری", "پاستا", "اسنک", "گریل",
                "مخلفات", "دسر",
            ]),
            (re.compile(r"(restaurant|رستوران|ایرانی|چلو|کباب|خورش)", re.I), [
                "کباب", "خورش", "پلو", "خوراک", "مخلفات", "دسر ایرانی", "شربت ایرانی", "آش", "حلیم", "سوپ",
                "دل و جگر", "طباخی",
            ]),
            (re.compile(r"(market|سوپر|grocery|خواربار|پت|pet)", re.I), [
                "برنج", "روغن", "چای", "کنسرو", "سس", "کمپوت",
                "قند و شکر و نبات", "چیپس و پفک", "بیسکوییت", "شکلات",
                "غذای خشک (پت)", "کنسرو و پوچ (پت)", "تشویقی و مکمل", "لوازم حیوانات",
            ]),
        ]

        self.fallback_keys: List[str] = [
            "کباب", "خورش", "پلو", "خوراک", "مخلفات",
            "پیتزا", "برگر", "ساندویچ", "سوخاری",
            "نوشیدنی گرم بر پایه قهوه", "نوشیدنی گرم غیر قهوه",
            "برنج", "کنسرو", "چیپس و پفک",
        ]

    def select_candidate_keys(
        self,
        batch_items: List[Dict[str, Any]],
        business_line: str,
        top_k: int,
        min_keys: int,
        include_arrow_targets: bool = True,
    ) -> List[str]:
        # Aggregate batch text (stable signal)
        parts: List[str] = []
        for it in batch_items:
            parts.append(str(it.get("title", "")))
            parts.append(str(it.get("desc", "")))
            parts.append(str(it.get("context_category", "")))
        batch_text = " ".join(parts)
        batch_tokens = tokenize(batch_text)

        scored: List[Tuple[int, str]] = []
        for k in self.all_keys:
            kt = self.key_tokens.get(k, set())
            bt = self.body_tokens.get(k, set())
            # Weight matches in the key more heavily
            score = 4 * len(batch_tokens & kt) + 1 * len(batch_tokens & bt)
            if score > 0:
                scored.append((score, k))
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected: List[str] = []
        used: Set[str] = set()

        # Business line hints
        bl = business_line or ""
        for pat, keys in self.business_line_hints:
            if pat.search(bl):
                for k in keys:
                    if k in self.valid_level_2_keys and k not in used:
                        selected.append(k)
                        used.add(k)

        # Top-K by score
        for _, k in scored:
            if len(selected) >= top_k:
                break
            if k not in used:
                selected.append(k)
                used.add(k)

        # Ensure minimum coverage
        target_min = max(min_keys, min(top_k, len(self.all_keys)))
        for k in self.fallback_keys:
            if len(selected) >= target_min:
                break
            if k in self.valid_level_2_keys and k not in used:
                selected.append(k)
                used.add(k)

        # Include valid arrow targets to make "follow →" possible
        if include_arrow_targets:
            expanded = list(selected)
            for k in selected:
                for tgt in self.arrow_targets.get(k, []):
                    if tgt in self.valid_level_2_keys and tgt not in used:
                        expanded.append(tgt)
                        used.add(tgt)
            selected = expanded

        return selected

    def subset_taxonomy(self, keys: List[str]) -> Dict[str, Any]:
        return {k: self.taxonomy[k] for k in keys if k in self.taxonomy}

# ============================ V14 BATCHING ("BEST OF ALL WORLDS") ============================

def iter_v14_batches(df_remaining: pd.DataFrame, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """
    V14 batching strategy:
    1) Partition by business_line (never mix)
    2) Within business_line, group by tf_code (vendor)
    3) Within vendor, sort by normalized item_title
    4) Pack vendor groups into batches until BATCH_SIZE; split vendor if needed
    """

    # Ensure required columns exist
    for col in ["business_line", "tf_code", "item_title", "item_id", "category_name", "item_description"]:
        if col not in df_remaining.columns:
            df_remaining[col] = ""

    work = df_remaining.copy()
    work["business_line"] = work["business_line"].fillna("").astype(str)
    work["tf_code"] = work["tf_code"].fillna("").astype(str)
    work["item_id"] = work["item_id"].fillna("").astype(str)
    work["item_title"] = work["item_title"].fillna("").astype(str)

    # Normalized title for stable locality
    work["_norm_title"] = work["item_title"].map(normalize_title_for_sort)

    # Stable sort (mergesort is stable)
    work.sort_values(["business_line", "tf_code", "_norm_title", "item_id"], inplace=True, kind="mergesort")

    current_bl: Optional[str] = None
    current_vendor: Optional[str] = None
    vendor_rows: List[Dict[str, Any]] = []

    def flush_vendor(bl: str, vendor: str, rows: List[Dict[str, Any]]) -> Iterable[Tuple[str, str, List[Dict[str, Any]]]]:
        if not rows:
            return
        # Split vendor if too large
        if len(rows) <= batch_size:
            yield (bl, vendor, rows)
        else:
            start = 0
            while start < len(rows):
                yield (bl, vendor, rows[start:start + batch_size])
                start += batch_size

    vendor_chunks: List[Tuple[str, str, List[Dict[str, Any]]]] = []

    for row in work.itertuples(index=False):
        bl = getattr(row, "business_line")
        vendor = getattr(row, "tf_code")
        d = row._asdict()
        d.pop("_norm_title", None)

        if current_bl is None:
            current_bl, current_vendor = bl, vendor

        if (bl != current_bl) or (vendor != current_vendor):
            vendor_chunks.extend(list(flush_vendor(current_bl, current_vendor, vendor_rows)))
            vendor_rows = []
            current_bl, current_vendor = bl, vendor

        vendor_rows.append(d)

    if vendor_rows:
        vendor_chunks.extend(list(flush_vendor(current_bl or "", current_vendor or "", vendor_rows)))

    # Pack vendor chunks into API batches (never mix business_line)
    batch: List[Dict[str, Any]] = []
    batch_bl: Optional[str] = None

    for bl, vendor, rows in vendor_chunks:
        if batch_bl is None:
            batch_bl = bl

        # business_line boundary flush
        if bl != batch_bl:
            if batch:
                yield batch
            batch = []
            batch_bl = bl

        # pack vendor chunk
        if len(batch) + len(rows) <= batch_size:
            batch.extend(rows)
        else:
            if batch:
                yield batch
                batch = []
            batch.extend(rows)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

# ============================ CLASSIFIER ENGINE (Top-K + UNKNOWN Second Pass) ============================

class ClassifierEngine:
    """
    Gemini 3 Flash Preview engine (original google.generativeai calling style):
    - Uses candidate subset taxonomy per batch (NOT full taxonomy).
    - First pass: top_k = CANDIDATE_TOP_K
    - Second pass: only UNKNOWN items, top_k = CANDIDATE_TOP_K_SECOND
    - level_1 is always derived from taxonomy (never trusted from model).
    """

    def __init__(self, taxonomy_path: str):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        self.taxonomy = taxonomy
        self.index = TaxonomyIndex(taxonomy)
        self.valid_level_2_keys: Set[str] = set(taxonomy.keys())

        self.model = genai.GenerativeModel(
            CONFIG["MODEL_NAME"],
            generation_config={
                "temperature": float(CONFIG["TEMPERATURE"]),
                "top_p": float(CONFIG["TOP_P"]),
                "top_k": int(CONFIG["TOP_K"]),
            },
        )

    def _fallback_results(self, products_for_prompt: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        return [{"id": p["id"], "level_1": "ERROR", "level_2": "ERROR", "reason": reason} for p in products_for_prompt]

    def _validate_and_normalize(
        self,
        products_for_prompt: List[Dict[str, Any]],
        parsed: Any,
        allowed_level2_set: Set[str],
        pass_tag: str,
    ) -> List[Dict[str, Any]]:
        """
        Normalize model output to strict format.
        Enforces:
        - exactly one result per input id
        - level_2 must be in allowed set, else -> ERROR
        - level_1 derived from taxonomy[level_2], else UNKNOWN/ERROR
        """
        id_order = [str(p["id"]) for p in products_for_prompt]

        if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
            parsed = parsed["results"]

        if not isinstance(parsed, list):
            return self._fallback_results(products_for_prompt, f"{pass_tag}_JSON_SHAPE_INVALID")

        result_map: Dict[str, Dict[str, Any]] = {}
        for r in parsed:
            if isinstance(r, dict) and "id" in r:
                result_map[str(r["id"])] = r

        out: List[Dict[str, Any]] = []
        for pid in id_order:
            r = result_map.get(pid)
            if not isinstance(r, dict):
                out.append({"id": pid, "level_1": "ERROR", "level_2": "ERROR", "reason": f"{pass_tag}_MISSING_ID"})
                continue

            l2 = str(r.get("level_2", "UNKNOWN")).strip()
            rsn = str(r.get("reason", "")).strip()

            # Strict: enforce allowed subset
            if l2 != "UNKNOWN" and l2 not in allowed_level2_set:
                out.append({
                    "id": pid,
                    "level_1": "ERROR",
                    "level_2": "ERROR",
                    "reason": f"{pass_tag}_INVALID_CANDIDATE_KEY: '{l2}' not in allowed set",
                })
                continue

            # Derive level_1 from taxonomy (never trust model for level_1)
            if l2 == "UNKNOWN":
                l1 = "UNKNOWN"
            else:
                l1 = str(self.taxonomy.get(l2, {}).get("level_1", "ERROR"))

            out.append({"id": pid, "level_1": l1, "level_2": l2, "reason": rsn})

        return out

    def _build_prompt(
        self,
        business_line: str,
        candidate_taxonomy: Dict[str, Any],
        products_for_prompt: List[Dict[str, Any]],
        allowed_level2: List[str],
        pass_instructions: str,
    ) -> str:
        tax_str = json.dumps(candidate_taxonomy, ensure_ascii=False, separators=(",", ":"))
        products_json = json.dumps(products_for_prompt, ensure_ascii=False, separators=(",", ":"))
        allowed_list = json.dumps(allowed_level2, ensure_ascii=False, separators=(",", ":"))

        prompt = (
            "You are a high-precision menu taxonomist.\n"
            "Task: For each item, select the best matching 'level_2' KEY from the provided candidate taxonomy.\n\n"
            f"BATCH CONTEXT:\n- business_line: {business_line}\n\n"
            "CANDIDATE TAXONOMY (JSON):\n"
            "Keys are Level_2 categories. Values include level_1, explanation, exclusions.\n"
            f"{tax_str}\n\n"
            "CRITICAL RULES:\n"
            "1) You MUST output level_2 as one of the allowed keys, otherwise use UNKNOWN.\n"
            "2) Apply exclusions strictly: if item matches an exclusion, follow the arrow (→) to the correct category.\n"
            "3) Prefer the most specific key that fits perfectly.\n"
            "4) Output must be strict JSON array only.\n\n"
            f"ALLOWED level_2 KEYS:\n{allowed_list}\n\n"
            f"{pass_instructions}\n\n"
            "INPUT ITEMS (JSON array):\n"
            f"{products_json}\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a JSON array. No markdown. Each element:\n"
            '[{"id":"...","level_2":"...","reason":"short reason; mention exclusion/arrow if applied"}, ...]'
        )
        return prompt

    def _call_model_with_retries(self, prompt: str, batch_info: str) -> Tuple[Any, str]:
        """
        Calls Gemini with retries and returns (parsed_json, raw_text).
        """
        last_err: Optional[str] = None

        for attempt in range(int(CONFIG["MAX_RETRIES"])):
            if shutdown_flag.is_set():
                raise RuntimeError(f"Shutdown triggered: {shutdown_flag.reason()}")

            try:
                rate_limiter.wait()
                response = self.model.generate_content(prompt)

                txt = getattr(response, "text", "") or ""
                if not txt.strip():
                    raise RuntimeError("Empty response from model")

                # token usage best-effort
                in_tokens = 0
                out_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                else:
                    in_tokens = max(1, len(prompt) // 3)
                    out_tokens = max(30, len(txt) // 4)

                cost_tracker.update(in_tokens, out_tokens)

                parsed = parse_json_strict(txt)
                return parsed, txt

            except Exception as e:
                last_err = str(e)
                logging.error(f"❌ Gemini call failed (attempt {attempt+1}/{CONFIG['MAX_RETRIES']}): {last_err[:240]}")

                le = last_err.lower()
                error_type = "UNKNOWN_ERROR"
                if "429" in le:
                    error_type = "429_QUOTA_EXCEEDED"
                elif "504" in le or "deadline" in le or "timeout" in le:
                    error_type = "504_TIMEOUT"
                elif "403" in le:
                    error_type = "403_FORBIDDEN"
                elif "500" in le or "502" in le or "503" in le:
                    error_type = "5XX_SERVER_ERROR"
                elif "failed to parse json" in le:
                    error_type = "JSON_PARSE_ERROR"

                error_notifier.notify_error(error_type, last_err, batch_info)

                if is_critical_gemini_error(last_err):
                    logging.critical(f"🚨 CRITICAL ERROR - Shutting down: {last_err}")
                    shutdown_flag.set(f"Critical Gemini error: {last_err}")
                    raise RuntimeError(shutdown_flag.reason())

                if attempt < int(CONFIG["MAX_RETRIES"]) - 1 and is_transient_gemini_error(last_err):
                    time.sleep(float(CONFIG["RETRY_DELAY_SEC"]))
                    continue

        raise RuntimeError(f"Gemini call failed after {CONFIG['MAX_RETRIES']} retries: {last_err or 'unknown'}")

    def _prepare_products_for_prompt(self, batch_data: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        # business_line should be homogeneous by batching strategy
        business_line = str(batch_data[0].get("business_line", "") or "")

        products_for_prompt: List[Dict[str, Any]] = []
        for item in batch_data:
            products_for_prompt.append({
                "id": normalize_item_id(item.get("item_id", "")),
                "title": _clean_text(item.get("item_title"), int(CONFIG["TRUNC_TITLE_CHARS"])),
                "desc": _clean_text(item.get("item_description"), int(CONFIG["TRUNC_DESC_CHARS"])),
                "context_category": _clean_text(item.get("category_name"), int(CONFIG["TRUNC_CTX_CHARS"])),
                "tf_code": _clean_text(item.get("tf_code"), 60),
                "business_line": _clean_text(item.get("business_line"), 60),
            })

        return business_line, products_for_prompt

    def _run_pass(
        self,
        pass_tag: str,
        business_line: str,
        products_for_prompt: List[Dict[str, Any]],
        top_k: int,
        min_keys: int,
        pass_instructions: str,
    ) -> List[Dict[str, Any]]:
        allowed_level2 = self.index.select_candidate_keys(
            batch_items=products_for_prompt,
            business_line=business_line,
            top_k=int(top_k),
            min_keys=int(min_keys),
            include_arrow_targets=bool(CONFIG["INCLUDE_ARROW_TARGETS"]),
        )
        allowed_set = set(allowed_level2)

        candidate_taxonomy = self.index.subset_taxonomy(allowed_level2)

        prompt = self._build_prompt(
            business_line=business_line,
            candidate_taxonomy=candidate_taxonomy,
            products_for_prompt=products_for_prompt,
            allowed_level2=allowed_level2,
            pass_instructions=pass_instructions,
        )

        batch_ids = [p["id"] for p in products_for_prompt]
        batch_info = f"{pass_tag} | bl={business_line} | IDs: {batch_ids[:3]}..."

        logging.info(f"➡️ {pass_tag} starting: bl='{business_line}' | n={len(batch_ids)} | topK={top_k}")
        parsed, _raw = self._call_model_with_retries(prompt, batch_info)
        normalized = self._validate_and_normalize(products_for_prompt, parsed, allowed_set, pass_tag)
        logging.info(f"✅ {pass_tag} completed: bl='{business_line}' | n={len(batch_ids)}")
        return normalized

    def classify_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if shutdown_flag.is_set():
            raise RuntimeError(f"Shutdown already triggered: {shutdown_flag.reason()}")

        business_line, products_for_prompt = self._prepare_products_for_prompt(batch_data)

        # --------------------
        # PASS 1 (Top-K small)
        # --------------------
        pass1_instructions = (
            "PASS 1 INSTRUCTION:\n"
            "- Choose the best fitting key from allowed keys.\n"
            "- If truly ambiguous, return UNKNOWN.\n"
            "- Keep reason very short."
        )

        pass1_results = self._run_pass(
            pass_tag="PASS1",
            business_line=business_line,
            products_for_prompt=products_for_prompt,
            top_k=int(CONFIG["CANDIDATE_TOP_K"]),
            min_keys=int(CONFIG["CANDIDATE_MIN_KEYS"]),
            pass_instructions=pass1_instructions,
        )

        # If second pass disabled, return immediately
        if not bool(CONFIG.get("SECOND_PASS_ENABLED", True)):
            return pass1_results

        # Identify UNKNOWN items only
        unknown_ids = [r["id"] for r in pass1_results if str(r.get("level_2")) == "UNKNOWN"]
        if not unknown_ids:
            return pass1_results

        # --------------------
        # PASS 2 (UNKNOWN only, Top-K bigger)
        # --------------------
        unknown_products = [p for p in products_for_prompt if p["id"] in set(unknown_ids)]

        # If somehow empty, return pass1
        if not unknown_products:
            return pass1_results

        pass2_instructions = (
            "PASS 2 INSTRUCTION (ONLY FOR UNKNOWN ITEMS):\n"
            "- Try harder to choose a valid key among allowed keys.\n"
            "- Use context_category + tf_code naming style + title clues.\n"
            "- Only return UNKNOWN if you are genuinely not confident.\n"
            "- Keep reason short; mention any exclusion arrow if applied.\n"
        )

        pass2_results = self._run_pass(
            pass_tag="PASS2",
            business_line=business_line,
            products_for_prompt=unknown_products,
            top_k=int(CONFIG["CANDIDATE_TOP_K_SECOND"]),
            min_keys=int(CONFIG["CANDIDATE_MIN_KEYS_SECOND"]),
            pass_instructions=pass2_instructions,
        )

        # Merge: replace UNKNOWN from pass1 if pass2 provides a valid non-UNKNOWN label
        pass2_map = {r["id"]: r for r in pass2_results if isinstance(r, dict)}
        merged: List[Dict[str, Any]] = []
        fixes = 0

        for r in pass1_results:
            if str(r.get("level_2")) != "UNKNOWN":
                merged.append(r)
                continue

            r2 = pass2_map.get(r["id"])
            if isinstance(r2, dict) and str(r2.get("level_2")) not in ("UNKNOWN", "ERROR", ""):
                # Replace with pass2 label
                merged.append({
                    "id": r["id"],
                    "level_1": r2.get("level_1", "ERROR"),
                    "level_2": r2.get("level_2", "ERROR"),
                    "reason": f"[2PASS] {r2.get('reason', '')}".strip(),
                })
                fixes += 1
            else:
                # keep UNKNOWN
                merged.append(r)

        stats.inc_second_pass(attempts=len(unknown_products), fixes=fixes)
        logging.info(f"🔁 Second-pass: attempted={len(unknown_products)} | fixed={fixes} | bl='{business_line}'")
        return merged

# ============================ REPORTING THREAD ============================

class MinuteReporter(threading.Thread):
    def __init__(self, stop_event: threading.Event, interval_sec: int):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.interval_sec = max(10, int(interval_sec))

    def run(self):
        while not self.stop_event.is_set():
            slept = 0
            while slept < self.interval_sec and not self.stop_event.is_set():
                time.sleep(1)
                slept += 1
            if self.stop_event.is_set():
                break

            if shutdown_flag.is_set():
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                break

            if bool(CONFIG.get("WATCHDOG_ENABLED", False)):
                current_processed = stats.snapshot()["processed"]
                should_stop, reason = rate_watchdog.check_rate(current_processed)
                if should_stop:
                    logging.critical(f"🚨 RATE WATCHDOG TRIGGERED: {reason}")
                    shutdown_flag.set(reason)
                    telegram.send(format_shutdown_message(reason))
                    break

            telegram.send(format_progress_message(stats.snapshot()))

# ============================ MAIN ============================

def main():
    print(f"=== AI PRODUCT CLASSIFIER V14.2 ({CONFIG['MODEL_NAME']}) ===")
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Taxonomy:    {TAXONOMY_FILE}")
    print(f"Workers:     {CONFIG['MAX_WORKERS']} | Batch size: {CONFIG['BATCH_SIZE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}")
    print(f"Retries:     {CONFIG['MAX_RETRIES']} | Retry delay: {CONFIG['RETRY_DELAY_SEC']}s")
    print(f"Temp:        {CONFIG['TEMPERATURE']} | top_p: {CONFIG['TOP_P']} | top_k: {CONFIG['TOP_K']}")
    print(f"Pass1 TopK:  {CONFIG['CANDIDATE_TOP_K']} (min {CONFIG['CANDIDATE_MIN_KEYS']})")
    print(f"Pass2 TopK:  {CONFIG['CANDIDATE_TOP_K_SECOND']} (min {CONFIG['CANDIDATE_MIN_KEYS_SECOND']}) | enabled={CONFIG['SECOND_PASS_ENABLED']}")
    print("")

    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, dtype={"item_id": "string"})
        df["item_id"] = df["item_id"].astype(str).str.replace(",", "", regex=False)

        # Ensure required columns exist
        for col in ["category_name", "item_title", "item_description", "business_line", "tf_code"]:
            if col not in df.columns:
                df[col] = ""

        total_rows = len(df)
        print(f"Total Products: {total_rows:,}")
    except Exception as e:
        print(f"Failed to read input file: {e}")
        telegram.send(f"[tf_menu] FAILED to read input file: {e}")
        return

    processed_ids = get_processed_ids(OUTPUT_FILE)
    print(f"Already Processed: {len(processed_ids):,} products")

    df_remaining = df[~df["item_id"].isin(processed_ids)].copy()
    remaining = len(df_remaining)

    if remaining == 0:
        print("All products processed! Script finished.")
        telegram.send("[tf_menu] All products already processed. Nothing to do.")
        return

    print(f"Remaining to Process: {remaining:,}")

    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=[
            "item_id", "tf_code", "business_line",
            "category_name", "item_title", "item_description",
            "level_1", "level_2", "reason"
        ]).to_csv(OUTPUT_FILE, index=False, encoding=CONFIG["OUTPUT_ENCODING"])

    try:
        engine = ClassifierEngine(TAXONOMY_FILE)
    except Exception as e:
        print(f"Failed to load taxonomy or init model: {e}")
        telegram.send(f"[tf_menu] FAILED to load taxonomy or init model: {e}")
        return

    stop_event = threading.Event()
    reporter = MinuteReporter(stop_event=stop_event, interval_sec=CONFIG["REPORT_EVERY_SECONDS"])
    reporter.start()

    telegram.send(
        f"[tf_menu] Started V14.2 - Stable Batching + TopK Taxonomy + UNKNOWN Second Pass\n"
        f"Model: {CONFIG['MODEL_NAME']}\n"
        f"Temp: {CONFIG['TEMPERATURE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}\n"
        f"Remaining: {remaining:,}\n"
        f"Workers: {CONFIG['MAX_WORKERS']} | Batch: {CONFIG['BATCH_SIZE']}\n"
        f"Pass1 TopK={CONFIG['CANDIDATE_TOP_K']} | Pass2 TopK={CONFIG['CANDIDATE_TOP_K_SECOND']} (UNKNOWN only)"
    )

    max_in_flight = max(1, int(CONFIG["MAX_WORKERS"]) * int(CONFIG["MAX_IN_FLIGHT_MULTIPLIER"]))

    batch_iter = iter(iter_v14_batches(df_remaining, int(CONFIG["BATCH_SIZE"])))

    def submit_next(executor, futures_map) -> bool:
        if shutdown_flag.is_set():
            return False
        try:
            batch = next(batch_iter)
        except StopIteration:
            return False
        fut = executor.submit(engine.classify_batch, batch)
        futures_map[fut] = batch
        return True

    total_batches_est = (remaining + int(CONFIG["BATCH_SIZE"]) - 1) // int(CONFIG["BATCH_SIZE"])

    logging.info(f"\n{'='*60}")
    logging.info(f"Starting V14.2 processing: {remaining:,} items (rough est ~{total_batches_est:,} batches)")
    logging.info(f"{'='*60}\n")

    batches_completed = 0

    try:
        with ThreadPoolExecutor(max_workers=int(CONFIG["MAX_WORKERS"])) as executor:
            futures_map: Dict[Any, List[Dict[str, Any]]] = {}

            for _ in range(max_in_flight):
                if not submit_next(executor, futures_map):
                    break

            while futures_map:
                if shutdown_flag.is_set():
                    break

                for fut in as_completed(list(futures_map.keys())):
                    batch_input = futures_map.pop(fut)
                    batches_completed += 1
                    progress_pct = (batches_completed / max(1, total_batches_est)) * 100
                    logging.info(f"📊 Progress: {batches_completed} batches done (~{progress_pct:.1f}%)")

                    if shutdown_flag.is_set():
                        break

                    try:
                        results = fut.result()
                        result_map = {str(r.get("id")): r for r in results if isinstance(r, dict)}

                        processed_rows: List[Dict[str, Any]] = []
                        for input_row in batch_input:
                            input_id = normalize_item_id(input_row.get("item_id", ""))
                            api_res = result_map.get(input_id, {})

                            row_out = {
                                "item_id": input_id,
                                "tf_code": input_row.get("tf_code"),
                                "business_line": input_row.get("business_line"),
                                "category_name": input_row.get("category_name"),
                                "item_title": input_row.get("item_title"),
                                "item_description": input_row.get("item_description"),
                                "level_1": api_res.get("level_1", "ERROR"),
                                "level_2": api_res.get("level_2", "ERROR"),
                                "reason": api_res.get("reason", ""),
                            }
                            processed_rows.append(row_out)

                        pd.DataFrame(processed_rows).to_csv(
                            OUTPUT_FILE, mode="a", header=False, index=False, encoding=CONFIG["OUTPUT_ENCODING"]
                        )
                        stats.update_from_rows(processed_rows)

                    except RuntimeError as e:
                        logging.critical(f"🚨 RuntimeError in main loop: {e}")
                        shutdown_flag.set(str(e))
                        break

                    except Exception as e:
                        msg = f"Error in main loop batch handling: {e}"
                        logging.error(f"❌ {msg}")
                        telegram.send(f"[tf_menu] {msg}")

                    if shutdown_flag.is_set():
                        break

                    while len(futures_map) < max_in_flight:
                        if not submit_next(executor, futures_map):
                            break

                    break

            if shutdown_flag.is_set():
                print("\n🚨 SHUTDOWN TRIGGERED. Cancelling remaining in-flight requests...")
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                for f in list(futures_map.keys()):
                    f.cancel()

    finally:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing loop ended. Batches completed: {batches_completed}")
        logging.info(f"{'='*60}\n")
        stop_event.set()

    if shutdown_flag.is_set():
        print("\n═══════════════════════════════════════════════════════════")
        print("🚨 APPLICATION STOPPED DUE TO CRITICAL GEMINI ERROR")
        print("Reason:", shutdown_flag.reason())
        print("Progress:", stats.snapshot())
        print("Cost:", cost_tracker.summary_str())
        print("═══════════════════════════════════════════════════════════")
        sys.exit(1)

    print("\nProcessing Complete.")
    snap = stats.snapshot()
    save_final_reports(snap["level1_counts"], snap["level12_counts"])
    telegram.send(format_final_message(snap))
    print(cost_tracker.summary_str())
    print("Saved final reports: final_level1_counts.csv, final_level2_counts.csv")


if __name__ == "__main__":
    main()
