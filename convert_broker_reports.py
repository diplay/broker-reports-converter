#!/usr/bin/env python3
"""Convert broker HTML reports into Snowball Income import CSV.

The script is designed for reports exported by Newton broker where
HTML tables contain sections like:
- 2. Состояние портфеля ценных бумаг (for ISIN mapping)
- 5.x Сделки с ценными бумагами (Buy/Sell)
- 8.1 Неторговые операции с ДС (dividends, taxes, cash movements)

Usage:
  python3 convert_broker_reports.py report1.html report2.html -o snowball.csv
  python3 convert_broker_reports.py /path/to/folder_with_reports
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

CSV_COLUMNS = [
    "Event",
    "Date",
    "Symbol",
    "Price",
    "Quantity",
    "Currency",
    "FeeTax",
    "Exchange",
    "NKD",
    "FeeCurrency",
    "DoNotAdjustCash",
    "Note",
]

DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
DEAL_ID_RE = re.compile(r"^[A-ZА-Я]-\d{6}-\d+$")
ISIN_RE = re.compile(r"RU[0-9A-Z]{10}")


@dataclass
class Cell:
    text: str
    attrs: Dict[str, str]


@dataclass
class Holding:
    name: str
    issuer: str
    reg_number: str
    isin: str
    qty_end: float

    @property
    def name_norm(self) -> str:
        return norm_key(self.name)

    @property
    def issuer_norm(self) -> str:
        return norm_key(self.issuer)

    @property
    def reg_norm(self) -> str:
        return norm_key(self.reg_number)

    @property
    def isin_norm(self) -> str:
        return norm_key(self.isin)

    @property
    def tokens(self) -> Set[str]:
        return tokens_for_match(" ".join([self.name, self.issuer, self.reg_number, self.isin]))


@dataclass
class EventRow:
    event: str
    date: str
    symbol: str
    price: str
    quantity: str
    currency: str
    feetax: str
    exchange: str = ""
    nkd: str = ""
    fee_currency: str = ""
    do_not_adjust_cash: str = ""
    note: str = ""
    order: int = 0

    def to_csv_dict(self) -> Dict[str, str]:
        return {
            "Event": self.event,
            "Date": self.date,
            "Symbol": self.symbol,
            "Price": self.price,
            "Quantity": self.quantity,
            "Currency": self.currency,
            "FeeTax": self.feetax,
            "Exchange": self.exchange,
            "NKD": self.nkd,
            "FeeCurrency": self.fee_currency,
            "DoNotAdjustCash": self.do_not_adjust_cash,
            "Note": self.note,
        }


class TableRowParser(HTMLParser):
    """Minimal HTML table parser using stdlib only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Cell]] = []
        self._in_tr = False
        self._in_cell = False
        self._current_text = ""
        self._current_attrs: Dict[str, str] = {}
        self._current_row: List[Cell] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = {k: (v or "") for k, v in attrs}
        if tag == "tr":
            self._in_tr = True
            self._current_row = []
            return
        if tag in {"td", "th"} and self._in_tr:
            self._in_cell = True
            self._current_text = ""
            self._current_attrs = attrs_dict

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_text += data

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            text = clean_text(self._current_text)
            self._current_row.append(Cell(text=text, attrs=self._current_attrs))
            self._in_cell = False
            self._current_text = ""
            self._current_attrs = {}
            return
        if tag == "tr" and self._in_tr:
            if self._current_row:
                self.rows.append(self._current_row)
            self._in_tr = False
            self._current_row = []


def clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def norm_key(value: str) -> str:
    value = value.upper().replace("Ё", "Е")
    value = re.sub(r"[^A-ZА-Я0-9]+", "", value)
    return value


def tokens_for_match(value: str) -> Set[str]:
    token_list = re.findall(r"[A-ZА-Я0-9]+", value.upper().replace("Ё", "Е"))
    return {tok for tok in token_list if len(tok) >= 4}


def parse_number(value: str) -> Optional[float]:
    value = clean_text(value)
    if not value:
        return None
    if DATE_RE.match(value) or TIME_RE.match(value):
        return None
    value = value.replace(" ", "").replace(",", ".")
    value = value.replace("%", "")
    if value in {"-", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_num(value: float) -> str:
    formatted = f"{value:.10f}".rstrip("0").rstrip(".")
    if formatted in {"", "-0"}:
        return "0"
    return formatted


def to_iso_date(value: str) -> str:
    return datetime.strptime(value, "%d.%m.%Y").strftime("%Y-%m-%d")


def normalize_currency(value: str) -> str:
    value = clean_text(value).upper()
    if value == "RUR":
        return "RUB"
    return value


def normalize_exchange(value: str) -> str:
    text = value.upper()
    if "МОСКОВСК" in text:
        return "MCX"
    if "NASDAQ" in text:
        return "NASDAQ"
    if "NYSE" in text:
        return "NYSE"
    if "ЛОНДОН" in text or "LSE" in text:
        return "LSE"
    if "ГОНКОН" in text or "HONG KONG" in text or re.search(r"\bHK\b", text):
        return "HK"
    return ""


def extract_exchange_from_trade_cells(texts: Sequence[str]) -> str:
    for value in reversed(texts):
        if not value:
            continue
        upper = value.upper()
        if "БИРЖ" in upper or "NASDAQ" in upper or "NYSE" in upper or "LSE" in upper:
            return normalize_exchange(value)
    return ""


def extract_holdings(rows: Sequence[Sequence[Cell]]) -> List[Holding]:
    result: Dict[str, Holding] = {}
    for row in rows:
        texts = [cell.text for cell in row]
        if len(texts) < 8:
            continue
        if texts[0].startswith("Наименование"):
            continue
        isin = texts[3]
        if not ISIN_RE.fullmatch(isin):
            continue
        qty_end = parse_number(texts[7])
        if qty_end is None:
            qty_end = 0.0
        holding = Holding(
            name=texts[0],
            issuer=texts[1],
            reg_number=texts[2],
            isin=isin,
            qty_end=qty_end,
        )
        result[holding.isin] = holding
    return list(result.values())


def find_holding_by_text(text: str, holdings: Sequence[Holding]) -> Optional[Holding]:
    if not text:
        return None

    isin_match = ISIN_RE.search(text)
    if isin_match:
        isin = isin_match.group(0)
        for holding in holdings:
            if holding.isin == isin:
                return holding

    comment_norm = norm_key(text)
    comment_tokens = tokens_for_match(text)

    best_score = 0
    best_holding: Optional[Holding] = None

    for holding in holdings:
        score = 0

        if holding.isin_norm and holding.isin_norm in comment_norm:
            score += 150
        if holding.reg_norm and holding.reg_norm in comment_norm:
            score += 100
        if holding.name_norm and holding.name_norm in comment_norm:
            score += 80
        if holding.issuer_norm and holding.issuer_norm in comment_norm:
            score += 50

        overlap = len(comment_tokens & holding.tokens)
        score += min(overlap, 6) * 5

        if score > best_score:
            best_score = score
            best_holding = holding

    if best_score >= 20:
        return best_holding
    return None


def extract_payout_per_security(comment: str) -> Optional[float]:
    match = re.search(r"Размер выплат на 1 ЦБ:\s*([0-9]+(?:[\.,][0-9]+)?)", comment)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def looks_like_trade_row(texts: Sequence[str]) -> bool:
    if len(texts) < 12:
        return False
    return (
        bool(DEAL_ID_RE.match(texts[0]))
        and bool(DATE_RE.match(texts[1]))
        and bool(TIME_RE.match(texts[2]))
        and texts[3] in {"Покупка", "Продажа"}
    )


def parse_extra_trade_fee(texts: Sequence[str]) -> float:
    """Best effort extraction of TS/stamp fees from section 5.1 / 5.10 rows."""
    extra = 0.0
    for idx in (12, 13, 18):
        if idx >= len(texts):
            continue
        value = texts[idx]
        if DATE_RE.match(value) or TIME_RE.match(value):
            continue
        num = parse_number(value)
        if num is None:
            continue
        # Additional fees in report are non-negative amounts.
        if num >= 0:
            extra += num
    return extra


def map_cash_operation(
    *,
    date_iso: str,
    op_type: str,
    amount: float,
    currency: str,
    comment: str,
    holdings: Sequence[Holding],
    order: int,
) -> EventRow:
    op_norm = norm_key(op_type)
    abs_amount = abs(amount)

    def mk(
        event: str,
        symbol: str,
        price: float,
        quantity: float,
        feetax: float,
        note_prefix: str,
        nkd: str = "",
        fee_currency: str = "",
    ) -> EventRow:
        return EventRow(
            event=event,
            date=date_iso,
            symbol=symbol,
            price=format_num(price),
            quantity=format_num(quantity),
            currency=currency,
            feetax=format_num(feetax),
            exchange="",
            nkd=nkd,
            fee_currency=fee_currency,
            do_not_adjust_cash="",
            note=f"{note_prefix}: {comment}" if comment else note_prefix,
            order=order,
        )

    if "ВЫВОДДС" in op_norm or "СПИСАНИЕДС" in op_norm:
        return mk("Cash_Out", currency, 1.0, abs_amount, 0.0, op_type)

    if "ПОПОЛНЕНИЕДС" in op_norm or "ВВОДДС" in op_norm or "ЗАЧИСЛЕНИЕДС" in op_norm:
        return mk("Cash_In", currency, 1.0, abs_amount, 0.0, op_type)

    if "НАЛОГ" in op_norm:
        return mk("Cash_Expense", currency, 1.0, abs_amount, 0.0, op_type)

    if "КОМИСС" in op_norm:
        return mk("Fee", "", 0.0, 0.0, abs_amount, op_type)

    if "ПОГАШЕНИЕНОМИНАЛ" in op_norm:
        holding = find_holding_by_text(comment, holdings)
        symbol = holding.isin if holding else ""
        price = extract_payout_per_security(comment)
        if price is None and holding and holding.qty_end > 0:
            price = abs_amount / holding.qty_end
        if price is None:
            price = 0.0
        return mk("Amortisation", symbol, price, abs_amount, 0.0, op_type)

    if "ПОЛНОЕПОГАШЕНИЕ" in op_norm:
        holding = find_holding_by_text(comment, holdings)
        symbol = holding.isin if holding else ""
        price = extract_payout_per_security(comment)
        if price is None:
            price = 0.0
        # For Repayment, Quantity should be count of bonds. If unknown, keep payout as qty fallback.
        quantity = holding.qty_end if holding else abs_amount
        return mk("Repayment", symbol, price, quantity, 0.0, op_type)

    if (
        "ДОХОДПОФИНАНСОВЫМИНСТРУМЕНТАМ" in op_norm
        or "ПОГАШЕНИЕКУПОН" in op_norm
        or "ДИВИДЕНД" in op_norm
    ):
        holding = find_holding_by_text(comment, holdings)
        symbol = holding.isin if holding else ""
        price = extract_payout_per_security(comment)
        if price is None and holding and holding.qty_end > 0:
            price = abs_amount / holding.qty_end
        if price is None:
            price = 0.0
        return mk("Dividend", symbol, price, abs_amount, 0.0, op_type)

    if "ВОЗВРАТНАЛОГ" in op_norm:
        return mk("Cash_Gain", currency, 1.0, abs_amount, 0.0, op_type)

    if amount >= 0:
        return mk("Cash_Gain", currency, 1.0, abs_amount, 0.0, op_type)
    return mk("Cash_Expense", currency, 1.0, abs_amount, 0.0, op_type)


def parse_report(path: Path, order_offset: int = 0) -> Tuple[List[EventRow], List[str], int]:
    parser = TableRowParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    rows = parser.rows

    holdings = extract_holdings(rows)
    events: List[EventRow] = []
    warnings: List[str] = []

    trade_seen_keys: Set[Tuple[str, str, str, str, str, str]] = set()

    in_trade_block = False
    in_cash_block = False
    in_security_ops_block = False
    current_instrument_header = ""
    row_order = order_offset

    for row in rows:
        texts = [cell.text for cell in row]
        if not texts:
            continue

        line = " ".join(t for t in texts if t)

        if "Номер сделки" in line and "Вид сделки" in line and "Цена одной ЦБ" in line:
            in_trade_block = True
            current_instrument_header = ""
            continue

        first_cell = texts[0] if texts else ""
        line_norm = norm_key(line)

        if first_cell.startswith("8.1") and "ДС" in line_norm:
            in_cash_block = True
            in_security_ops_block = False
            continue

        if first_cell.startswith("8.2"):
            in_cash_block = False
            in_security_ops_block = True
            continue

        if first_cell.startswith("9.") or first_cell.startswith("Дата формирования отчета"):
            in_cash_block = False
            in_security_ops_block = False

        if in_trade_block and len(texts) <= 2 and texts[0].startswith("8."):
            in_trade_block = False

        if in_trade_block:
            if len(texts) == 1 and texts[0] and not texts[0].startswith("Итого"):
                current_instrument_header = texts[0]
                continue

            if looks_like_trade_row(texts):
                deal_id = texts[0]
                date_iso = to_iso_date(texts[1])
                side = texts[3]
                event = "Buy" if side == "Покупка" else "Sell"

                price = parse_number(texts[4]) or 0.0
                quantity = parse_number(texts[6]) or 0.0
                nkd_value = parse_number(texts[7]) or 0.0

                price_currency = normalize_currency(texts[5])
                fee_currency = normalize_currency(texts[11]) if len(texts) > 11 else ""
                broker_fee = parse_number(texts[10]) or 0.0
                extra_fee = parse_extra_trade_fee(texts)
                fee_total = broker_fee + extra_fee

                exchange = extract_exchange_from_trade_cells(texts)
                holding = find_holding_by_text(current_instrument_header, holdings)
                symbol = holding.isin if holding else ""

                if not symbol:
                    # Fallback: best effort token extraction from header.
                    isin_match = ISIN_RE.search(current_instrument_header)
                    if isin_match:
                        symbol = isin_match.group(0)
                    else:
                        upper_tokens = re.findall(r"[A-Z]{2,10}", current_instrument_header.upper())
                        symbol = upper_tokens[0] if upper_tokens else ""

                fee_currency_for_csv = ""
                if fee_currency and fee_currency != price_currency:
                    fee_currency_for_csv = fee_currency

                trade_key = (
                    deal_id,
                    date_iso,
                    event,
                    format_num(price),
                    format_num(quantity),
                    symbol,
                )
                if trade_key in trade_seen_keys:
                    continue
                trade_seen_keys.add(trade_key)

                amount_value = texts[8] if len(texts) > 8 else ""
                amount_currency = normalize_currency(texts[9]) if len(texts) > 9 else ""

                events.append(
                    EventRow(
                        event=event,
                        date=date_iso,
                        symbol=symbol,
                        price=format_num(price),
                        quantity=format_num(quantity),
                        currency=price_currency,
                        feetax=format_num(fee_total),
                        exchange=exchange,
                        nkd=format_num(nkd_value) if nkd_value else "",
                        fee_currency=fee_currency_for_csv,
                        do_not_adjust_cash="",
                        note=(
                            f"Deal {deal_id}; amount {amount_value} {amount_currency}".strip()
                        ),
                        order=row_order,
                    )
                )
                row_order += 1

        # Cash operations block parser (8.1-like tables)
        if in_cash_block and len(texts) >= 5 and DATE_RE.match(texts[0]):
            amount = parse_number(texts[2])
            if amount is not None:
                op_type = texts[1]
                currency = normalize_currency(texts[3])
                comment = texts[4]

                events.append(
                    map_cash_operation(
                        date_iso=to_iso_date(texts[0]),
                        op_type=op_type,
                        amount=amount,
                        currency=currency,
                        comment=comment,
                        holdings=holdings,
                        order=row_order,
                    )
                )
                row_order += 1
                continue

        # Non-cash security transfers in 8.2 are detected but skipped.
        if in_security_ops_block and len(texts) >= 8 and DATE_RE.match(texts[0]):
            if texts[1] in {"Ввод ЦБ", "Вывод ЦБ", "Списание ЦБ", "Зачисление ЦБ"}:
                warnings.append(
                    f"{path.name}: skipped security transfer '{texts[1]}' on {texts[0]} ({texts[2]})"
                )

    return events, warnings, row_order


def resolve_input_files(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.html")))
            continue
        if path.is_file():
            files.append(path)
            continue
        raise FileNotFoundError(f"Input path not found: {item}")

    unique_files: List[Path] = []
    seen: Set[Path] = set()
    for file_path in sorted(files):
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)

    return unique_files


def write_csv(path: Path, events: Sequence[EventRow]) -> None:
    ordered = sorted(events, key=lambda row: (row.date, row.order))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow(row.to_csv_dict())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert broker HTML reports to Snowball Income CSV.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="HTML files and/or directories containing HTML reports",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="snowball_income_import.csv",
        help="Output CSV path (default: snowball_income_import.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only errors",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        files = resolve_input_files(args.inputs)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not files:
        print("No HTML files found", file=sys.stderr)
        return 2

    all_events: List[EventRow] = []
    all_warnings: List[str] = []

    order_counter = 0
    for file_path in files:
        events, warnings, order_counter = parse_report(file_path, order_counter)
        all_events.extend(events)
        all_warnings.extend(warnings)

    output_path = Path(args.output).expanduser().resolve()
    write_csv(output_path, all_events)

    if not args.quiet:
        event_stats: Dict[str, int] = {}
        for event in all_events:
            event_stats[event.event] = event_stats.get(event.event, 0) + 1

        print(f"Input files: {len(files)}")
        print(f"Output: {output_path}")
        print(f"Rows written: {len(all_events)}")
        print("Events:")
        for event_name in sorted(event_stats):
            print(f"  {event_name}: {event_stats[event_name]}")

        if all_warnings:
            print("Warnings:", file=sys.stderr)
            for warning in all_warnings:
                print(f"  - {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
