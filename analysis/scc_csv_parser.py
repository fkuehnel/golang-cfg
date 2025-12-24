#!/usr/bin/env python3
"""
Shared CSV parsing utilities for SCC analysis scripts.

Handles the tricky CSV format where function names may contain commas:
    func_name, num_blocks, num_kernels, scc_structure

Parses from the right since scc_structure always ends with ']'.
"""
import re
from pathlib import Path
from typing import Iterator, NamedTuple

# Parse from right: ", blocks, kernels, [...]" at end of line
_LINE_PATTERN = re.compile(r'^(.*),\s*(\d+)\s*,\s*(\d+)\s*,\s*(\[.*\])\s*$')

# For parsing SCC structure internals
_BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")
_BLOCK_TOKEN_RE = re.compile(r"b\d+")


class SCCRow(NamedTuple):
    """Parsed row from an SCC CSV file."""
    func: str
    blocks: int
    kernels: int
    structure: str


def parse_line(line: str) -> SCCRow | None:
    """Parse a single CSV line, handling commas in function names.
    
    Returns SCCRow or None if line doesn't match expected format.
    """
    m = _LINE_PATTERN.match(line.strip())
    if m:
        return SCCRow(
            func=m.group(1).strip(),
            blocks=int(m.group(2)),
            kernels=int(m.group(3)),
            structure=m.group(4),
        )
    return None


def iter_scc_rows(path: Path, pattern: str = "*_scc.csv") -> Iterator[tuple[Path, int, SCCRow]]:
    """Iterate over all matching CSV files, yielding (file, line_num, row).
    
    Args:
        path: Directory to search, or a single file
        pattern: Glob pattern for files (ignored if path is a file)
        
    Yields:
        (file_path, 1-indexed line number, SCCRow)
    """
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob(pattern))
    
    for f in files:
        with open(f, 'r', encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh, start=1):
                row = parse_line(line)
                if row:
                    yield f, i, row


def parse_scc_components(structure: str) -> list[list[str]]:
    """Parse SCC structure string into component lists.
    
    Example: '[[b1] [b6 b4 b10] [b11]]' -> [['b1'], ['b6', 'b4', 'b10'], ['b11']]
    """
    groups = _BRACKET_GROUP_RE.findall(structure)
    comps: list[list[str]] = []
    for g in groups:
        toks = _BLOCK_TOKEN_RE.findall(g)
        if toks:
            comps.append(toks)
    return comps


def parse_scc_sizes(structure: str) -> list[int]:
    """Parse SCC structure string into component sizes.
    
    Example: '[[b1] [b6 b4 b10] [b11]]' -> [1, 3, 1]
    """
    return [len(c) for c in parse_scc_components(structure)]


def get_scc_files(path: Path, pattern: str = "*_scc.csv") -> list[Path]:
    """Get list of matching SCC CSV files."""
    if path.is_file():
        return [path]
    files = sorted(path.glob(pattern))
    if not files:
        raise SystemExit(f"No files found in {path} matching {pattern}")
    return files
