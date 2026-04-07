import argparse
import ast
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from string import Formatter
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path("alerts.yml")
DEFAULT_STATE_PATH = Path(".alert_state.json")
DEFAULT_INTERVAL = "5m"
DEFAULT_PERIOD = "5d"
LOG_FORMAT = "%(levelname)s %(message)s"


@dataclass
class RuleResult:
    rule_id: str
    symbol_id: str
    yahoo_symbol: str
    candle_time: str
    message: str
    cooldown_minutes: int
    fingerprint: str


@dataclass
class ConditionResult:
    triggered: bool
    candle_time: str | None
    fingerprint: str


@dataclass
class SummaryResult:
    summary_id: str
    market_date: str
    message: str
    fingerprint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock alerts and notify Discord.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to alerts.yml")
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional override for the local state file used to suppress duplicates.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log alerts without posting to Discord.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("alerts.yml must contain a mapping at the top level.")

    return data


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rules": {}, "summaries": {}}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"State file is invalid: {path}")

    data.setdefault("rules", {})
    data.setdefault("summaries", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def get_discord_webhook(config: dict[str, Any]) -> str:
    discord = config.get("discord", {})
    if not isinstance(discord, dict):
        raise ValueError("discord must be a mapping in alerts.yml")

    if discord.get("webhook_url"):
        return str(discord["webhook_url"])

    env_name = str(discord.get("webhook_url_env", "DISCORD_WEBHOOK_URL"))
    webhook = os.getenv(env_name)
    if not webhook:
        raise ValueError(f"Discord webhook URL not configured. Set {env_name} or discord.webhook_url.")

    return webhook


def load_symbol_map(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    symbols = config.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("alerts.yml must define at least one symbol under symbols.")

    symbol_map: dict[str, dict[str, str]] = {}
    for item in symbols:
        if not isinstance(item, dict):
            raise ValueError("Each symbol entry must be a mapping.")

        symbol_id = str(item.get("id", "")).strip()
        yahoo_symbol = str(item.get("yahoo", "")).strip()
        if not symbol_id or not yahoo_symbol:
            raise ValueError("Each symbol entry must include id and yahoo.")

        symbol_map[symbol_id] = {
            "id": symbol_id,
            "yahoo": yahoo_symbol,
            "kind": str(item.get("kind", "")).strip(),
        }

    return symbol_map


def partial_format(template: str, context: dict[str, Any]) -> str:
    parts: list[str] = []
    for literal_text, field_name, format_spec, conversion in Formatter().parse(template):
        parts.append(literal_text)
        if field_name is None:
            continue

        if field_name in context:
            value = context[field_name]
            if conversion == "r":
                value = repr(value)
            elif conversion == "s":
                value = str(value)
            elif conversion == "a":
                value = ascii(value)

            parts.append(format(value, format_spec) if format_spec else str(value))
            continue

        placeholder = "{"
        placeholder += field_name
        if conversion:
            placeholder += f"!{conversion}"
        if format_spec:
            placeholder += f":{format_spec}"
        placeholder += "}"
        parts.append(placeholder)

    return "".join(parts)


def format_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return partial_format(value, context)
    if isinstance(value, list):
        return [format_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_template_value(item, context) for key, item in value.items()}
    return value


def expand_rule_templates(config: dict[str, Any]) -> list[dict[str, Any]]:
    rules = config.get("rules", [])
    if rules and not isinstance(rules, list):
        raise ValueError("rules must be a list in alerts.yml")

    templates = config.get("rule_templates", [])
    if templates and not isinstance(templates, list):
        raise ValueError("rule_templates must be a list in alerts.yml")

    if not templates:
        return rules

    symbol_map = load_symbol_map(config)
    symbol_groups = config.get("symbol_groups", {})
    if symbol_groups and not isinstance(symbol_groups, dict):
        raise ValueError("symbol_groups must be a mapping in alerts.yml")

    expanded_rules: list[dict[str, Any]] = list(rules)
    for template in templates:
        if not isinstance(template, dict):
            raise ValueError("Each rule template must be a mapping.")

        template_symbols = template.get("symbols")
        template_group = template.get("symbols_from_group")
        if template_symbols is not None and template_group is not None:
            raise ValueError("A rule template cannot define both symbols and symbols_from_group.")

        if template_group is not None:
            group_name = str(template_group)
            template_symbols = symbol_groups.get(group_name)
            if template_symbols is None:
                raise ValueError(f"Unknown symbol group in rule template: {group_name}")

        if not isinstance(template_symbols, list) or not template_symbols:
            raise ValueError("Each rule template must target at least one symbol.")

        template_body = {key: value for key, value in template.items() if key not in {"symbols", "symbols_from_group"}}
        for raw_symbol_id in template_symbols:
            symbol_id = str(raw_symbol_id).strip()
            symbol = symbol_map.get(symbol_id)
            if symbol is None:
                raise ValueError(f"Rule template references unknown symbol {symbol_id}.")

            context = {
                "symbol": symbol["id"],
                "symbol_id": symbol["id"],
                "symbol_lc": symbol["id"].lower(),
                "yahoo": symbol["yahoo"],
                "kind": symbol.get("kind", ""),
            }
            expanded_rule = format_template_value(template_body, context)
            if not isinstance(expanded_rule, dict):
                raise ValueError("Expanded rule template must produce a mapping.")

            expanded_rule["symbol"] = symbol_id
            expanded_rules.append(expanded_rule)

    return expanded_rules


def load_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    if data and not isinstance(data, dict):
        raise ValueError("data must be a mapping in alerts.yml")
    return data


def get_history_settings(config: dict[str, Any]) -> tuple[str, str]:
    data = load_data_config(config)
    interval = str(data.get("interval", DEFAULT_INTERVAL))
    period = str(data.get("period", DEFAULT_PERIOD))
    return interval, period


def fetch_symbol_histories(
    symbol_map: dict[str, dict[str, str]],
    interval: str,
    period: str,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol_id, symbol in symbol_map.items():
        histories[symbol_id] = fetch_history(symbol["yahoo"], interval=interval, period=period)
    return histories


def fetch_history(yahoo_symbol: str, interval: str, period: str) -> pd.DataFrame:
    logging.debug("Fetching %s history with interval=%s period=%s", yahoo_symbol, interval, period)
    history = yf.Ticker(yahoo_symbol).history(interval=interval, period=period, auto_adjust=False)
    if history.empty:
        raise ValueError(f"No price history returned for {yahoo_symbol}")

    history = history.rename(columns=str.lower)
    if "close" not in history.columns:
        raise ValueError(f"Price history for {yahoo_symbol} does not include close prices.")

    history = history.dropna(subset=["close"]).copy()
    if history.empty:
        raise ValueError(f"No usable close prices returned for {yahoo_symbol}")

    return history


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    RSI = Relative Strength Index.

    An RSI above 70 suggests a security is overbought (may drop), while below 30 suggests it is oversold (may rise).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, math.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def parse_metric_name(metric: str) -> tuple[str, int | None]:
    metric_name = metric.strip().lower()
    if metric_name in {"close", "price"}:
        return "close", None
    if metric_name.startswith("sma_"):
        return "sma", int(metric_name.split("_", maxsplit=1)[1])
    if metric_name.startswith("rsi_"):
        return "rsi", int(metric_name.split("_", maxsplit=1)[1])
    raise ValueError(f"Unsupported metric: {metric}")


def resolve_metric_series(
    metric_name: str,
    history: pd.DataFrame,
    metric_cache: dict[str, pd.Series],
) -> tuple[pd.Series, str]:
    if metric_name in metric_cache:
        return metric_cache[metric_name], metric_name

    metric_type, period = parse_metric_name(metric_name)
    close = history["close"]

    if metric_type == "close":
        series = close
    elif metric_type == "sma":
        series = close.rolling(window=period, min_periods=period).mean()
    elif metric_type == "rsi":
        series = compute_rsi(close, period)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    metric_cache[metric_name] = series
    return series, metric_name


def resolve_operand_series(
    operand: Any,
    history: pd.DataFrame,
    metric_cache: dict[str, pd.Series],
) -> tuple[pd.Series, str]:
    if isinstance(operand, (int, float)):
        value = float(operand)
        return pd.Series(value, index=history.index), str(value)

    operand_str = str(operand).strip()
    if not operand_str:
        raise ValueError("Rule operand cannot be empty.")

    try:
        value = float(operand_str)
        return pd.Series(value, index=history.index), operand_str
    except ValueError:
        pass

    if operand_str in metric_cache:
        return metric_cache[operand_str], operand_str

    expression = ast.parse(operand_str, mode="eval")

    def evaluate_expression(node: ast.AST) -> pd.Series:
        if isinstance(node, ast.Expression):
            return evaluate_expression(node.body)

        if isinstance(node, ast.Name):
            series, _ = resolve_metric_series(node.id, history, metric_cache)
            return series

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return pd.Series(float(node.value), index=history.index)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand_series = evaluate_expression(node.operand)
            return operand_series if isinstance(node.op, ast.UAdd) else -operand_series

        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div),
        ):
            left_series = evaluate_expression(node.left)
            right_series = evaluate_expression(node.right)
            if isinstance(node.op, ast.Add):
                return left_series + right_series
            if isinstance(node.op, ast.Sub):
                return left_series - right_series
            if isinstance(node.op, ast.Mult):
                return left_series * right_series
            return left_series / right_series

        raise ValueError(f"Unsupported expression: {operand_str}")

    series = evaluate_expression(expression)
    metric_cache[operand_str] = series
    return series, operand_str


def evaluate_condition(op: str, left: pd.Series, right: pd.Series) -> tuple[bool, pd.DataFrame]:
    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if aligned.empty:
        return False, aligned

    op_name = op.strip().lower()
    if op_name == "gt":
        return bool(aligned.iloc[-1]["left"] > aligned.iloc[-1]["right"]), aligned
    if op_name == "gte":
        return bool(aligned.iloc[-1]["left"] >= aligned.iloc[-1]["right"]), aligned
    if op_name == "lt":
        return bool(aligned.iloc[-1]["left"] < aligned.iloc[-1]["right"]), aligned
    if op_name == "lte":
        return bool(aligned.iloc[-1]["left"] <= aligned.iloc[-1]["right"]), aligned

    if len(aligned) < 2:
        return False, aligned

    prev_left = aligned.iloc[-2]["left"]
    prev_right = aligned.iloc[-2]["right"]
    last_left = aligned.iloc[-1]["left"]
    last_right = aligned.iloc[-1]["right"]

    if op_name == "crosses_above":
        return bool(prev_left <= prev_right and last_left > last_right), aligned
    if op_name == "crosses_below":
        return bool(prev_left >= prev_right and last_left < last_right), aligned

    raise ValueError(f"Unsupported operator: {op}")


def describe_condition(condition: dict[str, Any]) -> str:
    if "all" in condition:
        children = condition["all"]
        return " and ".join(describe_condition(child) for child in children)
    if "any" in condition:
        children = condition["any"]
        return " or ".join(describe_condition(child) for child in children)

    return f"{condition.get('left')} {condition.get('op')} {condition.get('right')}"


def evaluate_rule_condition(
    condition: dict[str, Any],
    history: pd.DataFrame,
    metric_cache: dict[str, pd.Series],
) -> ConditionResult:
    if not isinstance(condition, dict):
        raise ValueError("Each condition must be a mapping.")

    if "all" in condition or "any" in condition:
        if "all" in condition and "any" in condition:
            raise ValueError("A condition cannot define both all and any at the same level.")

        mode = "all" if "all" in condition else "any"
        children = condition[mode]
        if not isinstance(children, list) or not children:
            raise ValueError(f"{mode} conditions must be a non-empty list.")

        child_results = [evaluate_rule_condition(child, history, metric_cache) for child in children]
        triggered = all(result.triggered for result in child_results) if mode == "all" else any(
            result.triggered for result in child_results
        )
        relevant = child_results if mode == "all" else [result for result in child_results if result.triggered]
        candle_times = [result.candle_time for result in relevant if result.candle_time]
        candle_time = max(candle_times) if candle_times else None
        fingerprint = (
            f"{mode}("
            + "|".join(f"{int(result.triggered)}:{result.fingerprint}" for result in child_results)
            + ")"
        )
        return ConditionResult(triggered=triggered, candle_time=candle_time, fingerprint=fingerprint)

    left_series, left_label = resolve_operand_series(condition.get("left"), history, metric_cache)
    right_series, right_label = resolve_operand_series(condition.get("right"), history, metric_cache)
    op = str(condition.get("op", ""))
    triggered, aligned = evaluate_condition(op, left_series, right_series)
    if aligned.empty:
        return ConditionResult(
            triggered=False,
            candle_time=None,
            fingerprint=f"{left_label}:{op}:{right_label}:empty",
        )

    candle_time = str(aligned.index[-1].isoformat())
    left_value = float(aligned.iloc[-1]["left"])
    right_value = float(aligned.iloc[-1]["right"])
    fingerprint = (
        f"{left_label}:{op}:{right_label}:{candle_time}:{left_value:.4f}:{right_value:.4f}"
    )
    return ConditionResult(triggered=triggered, candle_time=candle_time, fingerprint=fingerprint)


def extract_format_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.add(field_name)
    return fields


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def parse_clock_time(value: str, label: str) -> dt_time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"{label} must use HH:MM 24-hour format.") from exc


def should_send_alert(
    state: dict[str, Any],
    rule_id: str,
    fingerprint: str,
    cooldown_minutes: int,
    now: datetime,
) -> bool:
    previous = state["rules"].get(rule_id, {})
    if previous.get("last_fingerprint") == fingerprint:
        return False

    last_sent_at = previous.get("last_sent_at")
    if not last_sent_at:
        return True

    last_dt = datetime.fromisoformat(last_sent_at)
    return now - last_dt >= timedelta(minutes=cooldown_minutes)


def should_send_summary(state: dict[str, Any], summary_id: str, market_date: str) -> bool:
    previous = state["summaries"].get(summary_id, {})
    return previous.get("last_market_date") != market_date


def render_message(
    rule: dict[str, Any],
    symbol_id: str,
    yahoo_symbol: str,
    candle_time: str,
    history: pd.DataFrame,
    metric_cache: dict[str, pd.Series],
    condition_summary: str,
) -> str:
    template = str(
        rule.get("message") or "{symbol} triggered {rule_id}: {condition} at {candle_time}"
    )
    fields = extract_format_fields(template)

    context: dict[str, Any] = {
        "symbol": symbol_id,
        "yahoo_symbol": yahoo_symbol,
        "rule_id": str(rule["id"]),
        "candle_time": candle_time,
        "condition": condition_summary,
    }

    for field in fields:
        if field in context:
            continue
        try:
            series, _ = resolve_operand_series(field, history, metric_cache)
        except ValueError:
            continue

        aligned = series.dropna()
        if not aligned.empty:
            context[field] = float(aligned.iloc[-1])

    return template.format(**context)


def resolve_symbol_list(
    selectors: list[Any] | None,
    group_names: list[Any] | None,
    symbol_map: dict[str, dict[str, str]],
    symbol_groups: dict[str, Any],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    if selectors is not None:
        if not isinstance(selectors, list) or not selectors:
            raise ValueError("symbols must be a non-empty list when provided.")
        for raw_symbol_id in selectors:
            symbol_id = str(raw_symbol_id).strip()
            if symbol_id not in symbol_map:
                raise ValueError(f"Unknown symbol referenced: {symbol_id}")
            if symbol_id not in seen:
                selected.append(symbol_id)
                seen.add(symbol_id)

    if group_names is not None:
        if not isinstance(group_names, list) or not group_names:
            raise ValueError("symbols_from_groups must be a non-empty list when provided.")
        for raw_group_name in group_names:
            group_name = str(raw_group_name).strip()
            group_symbols = symbol_groups.get(group_name)
            if not isinstance(group_symbols, list) or not group_symbols:
                raise ValueError(f"Unknown or empty symbol group referenced: {group_name}")
            for raw_symbol_id in group_symbols:
                symbol_id = str(raw_symbol_id).strip()
                if symbol_id not in symbol_map:
                    raise ValueError(f"Unknown symbol referenced from group {group_name}: {symbol_id}")
                if symbol_id not in seen:
                    selected.append(symbol_id)
                    seen.add(symbol_id)

    if selected:
        return selected

    return list(symbol_map.keys())


def get_market_session(history: pd.DataFrame, market_tz: ZoneInfo, market_date: datetime.date) -> pd.DataFrame:
    localized = history.copy()
    if localized.index.tz is None:
        localized.index = localized.index.tz_localize(timezone.utc)
    localized.index = localized.index.tz_convert(market_tz)
    session_mask = localized.index.date == market_date
    return localized.loc[session_mask]


def format_summary_message(
    summary_title: str,
    line_template: str,
    summary_type: str,
    market_date: str,
    tz_name: str,
    rows: list[dict[str, Any]],
    missing_symbols: list[str],
) -> str:
    title = partial_format(
        summary_title,
        {
            "summary_type": summary_type,
            "market_date": market_date,
            "timezone": tz_name,
        },
    )
    lines = [title]
    for row in rows:
        lines.append(partial_format(line_template, row))

    if missing_symbols:
        lines.append(f"Missing data: {', '.join(missing_symbols)}")

    return "\n".join(lines)


def collect_summary_alerts(
    config: dict[str, Any],
    symbol_map: dict[str, dict[str, str]],
    symbol_histories: dict[str, pd.DataFrame],
    now: datetime,
) -> list[SummaryResult]:
    market_summaries = config.get("market_summaries", {})
    if not market_summaries:
        return []
    if not isinstance(market_summaries, dict):
        raise ValueError("market_summaries must be a mapping in alerts.yml")

    symbol_groups = config.get("symbol_groups", {})
    if symbol_groups and not isinstance(symbol_groups, dict):
        raise ValueError("symbol_groups must be a mapping in alerts.yml")

    tz_name = str(market_summaries.get("timezone", "America/Toronto"))
    market_tz = ZoneInfo(tz_name)
    now_local = now.astimezone(market_tz)
    market_date = now_local.date()

    summary_defaults = {
        "open": {
            "id": "market_open_prices",
            "send_after": "09:35",
            "title": "Market open prices for {market_date}",
            "line_template": "- {symbol}: {price:.2f}",
        },
        "close": {
            "id": "market_close_prices",
            "send_after": "16:05",
            "title": "Market close prices for {market_date}",
            "line_template": "- {symbol}: {price:.2f}",
        },
    }

    results: list[SummaryResult] = []
    for summary_type in ("open", "close"):
        summary_config = market_summaries.get(summary_type, {})
        if summary_config is None:
            continue
        if not isinstance(summary_config, dict):
            raise ValueError(f"market_summaries.{summary_type} must be a mapping.")

        if not summary_config.get("enabled", True):
            continue

        default_config = summary_defaults[summary_type]
        send_after = parse_clock_time(
            str(summary_config.get("send_after", default_config["send_after"])),
            f"market_summaries.{summary_type}.send_after",
        )
        if now_local.time() < send_after:
            continue

        symbol_ids = resolve_symbol_list(
            selectors=summary_config.get("symbols"),
            group_names=summary_config.get("symbols_from_groups"),
            symbol_map=symbol_map,
            symbol_groups=symbol_groups,
        )

        rows: list[dict[str, Any]] = []
        missing_symbols: list[str] = []
        fingerprint_parts: list[str] = []
        for symbol_id in symbol_ids:
            symbol = symbol_map[symbol_id]
            session = get_market_session(symbol_histories[symbol_id], market_tz, market_date)
            if session.empty:
                missing_symbols.append(symbol_id)
                continue

            if summary_type == "open":
                price = float(session.iloc[0]["open"] if "open" in session.columns else session.iloc[0]["close"])
                candle_time = session.index[0]
            else:
                price = float(session.iloc[-1]["close"])
                candle_time = session.index[-1]

            rows.append(
                {
                    "symbol": symbol_id,
                    "yahoo_symbol": symbol["yahoo"],
                    "kind": symbol.get("kind", ""),
                    "price": price,
                    "candle_time": candle_time.isoformat(),
                }
            )
            fingerprint_parts.append(f"{symbol_id}:{price:.4f}:{candle_time.isoformat()}")

        if not rows:
            continue

        summary_id = str(summary_config.get("id", default_config["id"]))
        title = str(summary_config.get("title", default_config["title"]))
        line_template = str(summary_config.get("line_template", default_config["line_template"]))
        market_date_str = market_date.isoformat()
        message = format_summary_message(
            summary_title=title,
            line_template=line_template,
            summary_type=summary_type,
            market_date=market_date_str,
            tz_name=tz_name,
            rows=rows,
            missing_symbols=missing_symbols,
        )
        fingerprint = f"{summary_type}:{market_date_str}:{'|'.join(fingerprint_parts)}:missing={','.join(missing_symbols)}"
        results.append(
            SummaryResult(
                summary_id=summary_id,
                market_date=market_date_str,
                message=message,
                fingerprint=fingerprint,
            )
        )

    return results


def build_rule_result(
    rule: dict[str, Any],
    symbol: dict[str, str],
    history: pd.DataFrame,
) -> RuleResult | None:
    metric_cache: dict[str, pd.Series] = {}
    condition = rule.get("condition", {})
    if not isinstance(condition, dict):
        raise ValueError(f"Rule {rule.get('id')} must define a condition mapping.")

    condition_result = evaluate_rule_condition(condition, history, metric_cache)
    if not condition_result.triggered or not condition_result.candle_time:
        return None

    message = render_message(
        rule=rule,
        symbol_id=symbol["id"],
        yahoo_symbol=symbol["yahoo"],
        candle_time=condition_result.candle_time,
        history=history,
        metric_cache=metric_cache,
        condition_summary=describe_condition(condition),
    )
    fingerprint = f"{rule['id']}::{condition_result.fingerprint}"
    cooldown_minutes = int(rule.get("cooldown_minutes", 60))

    return RuleResult(
        rule_id=str(rule["id"]),
        symbol_id=symbol["id"],
        yahoo_symbol=symbol["yahoo"],
        candle_time=condition_result.candle_time,
        message=message,
        cooldown_minutes=cooldown_minutes,
        fingerprint=fingerprint,
    )


def collect_rule_alerts(
    config: dict[str, Any],
    symbol_map: dict[str, dict[str, str]],
    symbol_histories: dict[str, pd.DataFrame],
) -> list[RuleResult]:
    rules = expand_rule_templates(config)
    if not isinstance(rules, list):
        raise ValueError("alerts.yml rules must expand to a list.")
    if not rules:
        return []

    alerts: list[RuleResult] = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each rule entry must be a mapping.")

        rule_id = str(rule.get("id", "")).strip()
        symbol_id = str(rule.get("symbol", "")).strip()
        if not rule_id or not symbol_id:
            raise ValueError("Each rule must include id and symbol.")

        symbol = symbol_map.get(symbol_id)
        if symbol is None:
            raise ValueError(f"Rule {rule_id} references unknown symbol {symbol_id}.")

        result = build_rule_result(rule, symbol, symbol_histories[symbol_id])
        if result is not None:
            alerts.append(result)

    return alerts


def post_to_discord(webhook_url: str, content: str) -> None:
    response = requests.post(webhook_url, json={"content": content}, timeout=20)
    response.raise_for_status()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    load_dotenv()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    data = load_data_config(config)

    state_path = Path(args.state_file) if args.state_file else Path(data.get("state_file", DEFAULT_STATE_PATH))
    state = load_state(state_path)
    webhook_url = "" if args.dry_run else get_discord_webhook(config)
    now = datetime.now(timezone.utc)

    symbol_map = load_symbol_map(config)
    interval, period = get_history_settings(config)
    symbol_histories = fetch_symbol_histories(symbol_map, interval=interval, period=period)
    summary_alerts = collect_summary_alerts(config, symbol_map, symbol_histories, now)
    rule_alerts = collect_rule_alerts(config, symbol_map, symbol_histories)

    if not summary_alerts and not rule_alerts:
        save_state(state_path, state)
        logging.info("No alerts fired.")
        return 0

    sent_count = 0
    for summary in summary_alerts:
        if not should_send_summary(state, summary.summary_id, summary.market_date):
            logging.info("Skipping already-sent summary for %s on %s", summary.summary_id, summary.market_date)
            continue

        if args.dry_run:
            logging.info("[dry-run] %s", summary.message)
        else:
            post_to_discord(webhook_url, summary.message)
            logging.info("Sent summary %s", summary.summary_id)

        state["summaries"][summary.summary_id] = {
            "last_fingerprint": summary.fingerprint,
            "last_market_date": summary.market_date,
            "last_sent_at": isoformat_utc(now),
        }
        sent_count += 1

    for alert in rule_alerts:
        if not should_send_alert(state, alert.rule_id, alert.fingerprint, alert.cooldown_minutes, now):
            logging.info("Skipping duplicate or cooling-down alert for %s", alert.rule_id)
            continue

        if args.dry_run:
            logging.info("[dry-run] %s", alert.message)
        else:
            post_to_discord(webhook_url, alert.message)
            logging.info("Sent alert for %s", alert.rule_id)

        state["rules"][alert.rule_id] = {
            "last_candle_time": alert.candle_time,
            "last_fingerprint": alert.fingerprint,
            "last_sent_at": isoformat_utc(now),
        }
        sent_count += 1

    save_state(state_path, state)
    logging.info(
        "Processed %s summary alert(s) and %s rule alert(s), sent %s.",
        len(summary_alerts),
        len(rule_alerts),
        sent_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
