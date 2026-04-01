from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TaguchiBlock:
    title: str
    param_levels: Dict[str, Tuple[float, float, float]]
    table: List[List[int]]


@dataclass(frozen=True)
class ParsedSheet:
    sheet_name: str
    blocks: Dict[str, TaguchiBlock]  # keys: "base", "adaptive"


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if x != x:  # NaN
            return ""
    except Exception:
        pass
    return str(x)


def _parse_table_from_column(col_cells: List[str]) -> List[List[int]]:
    """
    Parse a Taguchi table that is visually split across rows in one column.
    Expected something like:
      [[1,1,1,1,1,1],
       [1,1,1,2,2,2],
       ...
       [3,3,2,3,2,1]]
    """
    start = None
    for i, s in enumerate(col_cells):
        if "[[" in s:
            start = i
            break
    if start is None:
        raise ValueError("Could not find table start '[['")

    parts: List[str] = []
    for s in col_cells[start:]:
        s = s.strip()
        if not s:
            continue
        if "[" not in s and "]" not in s:
            # likely unrelated text
            continue
        parts.append(s)
        if "]]" in s:
            break

    raw = " ".join(parts)
    # Some cells may omit the closing brackets due to truncation/formatting.
    if raw.count("[") > raw.count("]"):
        raw = raw + ("]" * (raw.count("[") - raw.count("]")))
    # Ensure it's a Python literal.
    raw = raw.replace("\u00a0", " ")
    tbl = ast.literal_eval(raw)
    if not isinstance(tbl, list) or not tbl or not isinstance(tbl[0], list):
        raise ValueError("Parsed table has unexpected shape")
    return [[int(v) for v in row] for row in tbl]


def parse_sheet_blocks(excel_path: Path, sheet_name: str) -> ParsedSheet:
    """
    Extract parameter levels + Taguchi tables from a non-structured Excel.

    Assumptions (fit `Nikita_AL-GEA.xlsx` layout):
    - Each sheet contains two blocks: base (left) and adaptive (right)
    - Each block starts with a param row 'Npop'
    - Param rows: name in first col of block, then 3 numeric levels in next 3 cols
    - Taguchi table is embedded as a multi-row string starting with '[[' in the last col of the block
    """
    import pandas as pd  # local import so unit tests don't require pandas

    df = pd.ExcelFile(excel_path).parse(sheet_name, header=None)

    def parse_block(block_label: str, anchor_col_guess: int) -> TaguchiBlock:
        # find the 'Npop' row in the neighborhood of anchor_col_guess
        anchor_row = None
        anchor_col = None
        for r in range(min(30, df.shape[0])):
            for c in range(max(0, anchor_col_guess - 2), min(df.shape[1], anchor_col_guess + 4)):
                if _as_str(df.iat[r, c]).strip() == "Npop":
                    anchor_row = r
                    anchor_col = c
                    break
            if anchor_row is not None:
                break
        if anchor_row is None or anchor_col is None:
            raise ValueError(f"Could not locate Npop for block '{block_label}'")

        param_levels: Dict[str, Tuple[float, float, float]] = {}
        r = anchor_row
        while r < df.shape[0]:
            name = _as_str(df.iat[r, anchor_col]).strip()
            if not name:
                r += 1
                continue
            if "Taguchi Table" in name:
                break
            # Expect 3 levels in next cols
            vals = []
            for cc in range(anchor_col + 1, anchor_col + 4):
                v = df.iat[r, cc]
                if v is None:
                    vals.append(None)
                else:
                    vals.append(v)
            if all(v == "" or v is None or (isinstance(v, float) and v != v) for v in vals):
                # not a param row
                r += 1
                continue
            try:
                a, b, c = (float(vals[0]), float(vals[1]), float(vals[2]))
            except Exception as e:
                raise ValueError(f"Bad levels for {name} in block {block_label}: {vals}") from e
            param_levels[name] = (a, b, c)
            r += 1

        # Find a column that contains the '[[' table start near this block
        table_col = None
        for c in range(anchor_col, min(df.shape[1], anchor_col + 8)):
            col_strs = [_as_str(x) for x in df.iloc[:, c].tolist()]
            if any("[[" in s for s in col_strs):
                table_col = c
                break
        if table_col is None:
            raise ValueError(f"Could not locate Taguchi table text for block '{block_label}'")

        col_cells = [_as_str(x) for x in df.iloc[:, table_col].tolist()]
        table = _parse_table_from_column(col_cells)
        return TaguchiBlock(title=block_label, param_levels=param_levels, table=table)

    # In this Excel the base block is on the left, adaptive on the right
    base = parse_block("base", anchor_col_guess=0)
    adaptive = parse_block("adaptive", anchor_col_guess=8)
    return ParsedSheet(sheet_name=sheet_name, blocks={"base": base, "adaptive": adaptive})

