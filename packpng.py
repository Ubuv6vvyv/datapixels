#!/usr/bin/env python3
"""
LLMPv2 Termux Packer — 1x1 Native Pixel Scale + Sandbox-Safe fitz/NumPy Decoder
- Default 1x1 RGB pixel mapping (~11.25 MiB compressed / ~45 MB raw per 2048x2048 frame)
  for WhatsApp Document PDF 1.4 /FlateDecode and lossless PNG transport.
- Automatic Virtual Row-Chunking for oversized CSV, TSV, JSONL, NDJSON, and LOG files
  (> --block-kb): splits cleanly on newline boundaries in O(64 KB) RAM and prepends
  CSV/TSV column headers to every chunk so each INDEX_XX block is standalone.
- Zero-blocked-import sandbox decoder & embedded `00_scan.py`: uses ONLY `fitz`,
  `numpy`, and `json` (bypassing sandbox `zlib`/`re`/`struct`/`glob` blocks via
  in-memory PDF `/FlateDecode` `xref_stream` decompression) to run generic deduplication,
  Shannon entropy scanning, information-weight ranking, and tabular profiling.
- Every generated .pdf / .png part is 100% SELF-CONTAINED (independent zlib
  stream + local TOC), so any single part decodes standalone.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import gc
import hashlib
import json
import math
import re
import resource
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# CONFIGURATION TOGGLES
# ==============================================================================
DEFAULT_EMBED_SCANNER = True    # Set True to ALWAYS embed 00_scan.py without flags
SCANNER_FILENAME = "00_scan.py"
HEADER_H = 128
DEFAULT_SCALE = 1                # 1x1 native RGB pixels (3 bytes per pixel)
DEFAULT_MAX_EDGE = 2048          # 2048x2048 max canvas -> 2048x1920 payload = 11,796,480 B
DEFAULT_BLOCK_KB = 500           # Max raw KB per logical LLM working-set block
DEFAULT_MAX_RAW_MB = 100         # Safety ceiling for runaway directory scans
STREAM_CHUNK = 65_536            # 64 KB streaming I/O buffer
OUTLINE_PEEK_BYTES = 4_096       # Peek first 4 KB for headings/keys
COMPACT_TOC_THRESHOLD = 250      # Omit verbose per-file outlines above this count

CHUNKABLE_EXTS = {".csv", ".tsv", ".jsonl", ".ndjson", ".log"}
TABULAR_EXTS = {".csv", ".tsv"}

DEFAULT_EXTS = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".ndjson", ".csv", ".tsv",
    ".yaml", ".yml", ".xml", ".html", ".htm", ".py", ".js", ".ts",
    ".jsx", ".tsx", ".java", ".kt", ".kts", ".c", ".cpp", ".cc", ".h",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".dart",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql", ".r",
    ".log", ".ini", ".cfg", ".conf", ".toml", ".env", ".properties",
    ".gradle", ".rst", ".tex",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".idea", ".gradle", "build", "dist", "Android",
}


def make_scanner_code(pdf_names: list[str]) -> str:
    """
    Generates self-contained 00_scan.py using ONLY `fitz`, `numpy`, and `json`
    so it runs cleanly inside locked-down LLM sandboxes (Meta AI / Gemini)
    where `zlib`, `re`, `struct`, `glob`, and `hashlib` imports are blocked.
    Performs generic deduplication, Shannon entropy measurement, information-density
    weighting, and tabular CSV/TSV column profiling.
    """
    pdf_list_literal = json.dumps(pdf_names)
    return (
        '#!/usr/bin/env python3\n'
        '"""\n'
        '00_scan.py — Sandbox-Safe LLMPv2 Entropy, Deduplication & Weight Analyzer\n'
        'Uses ONLY `fitz`, `numpy`, and `json` (no `zlib`, `re`, `struct`, `glob`, or `hashlib`).\n'
        '"""\n'
        'import fitz, json\n'
        'import numpy as np\n\n'
        f'DEFAULT_PDFS = {pdf_list_literal}\n\n\n'
        'def _inflate_via_fitz(comp_bytes):\n'
        '    hdr = b"%PDF-1.4\\n"\n'
        '    o1 = b"1 0 obj\\n<< /Type /Catalog /Pages 2 0 R >>\\nendobj\\n"\n'
        '    o2 = b"2 0 obj\\n<< /Type /Pages /Kids [] /Count 0 >>\\nendobj\\n"\n'
        '    off1 = len(hdr)\n'
        '    off2 = off1 + len(o1)\n'
        '    off3 = off2 + len(o2)\n'
        '    o3 = (\n'
        '        f"3 0 obj\\n<< /Length {len(comp_bytes)} /Filter /FlateDecode >>\\nstream\\n".encode("ascii")\n'
        '        + comp_bytes\n'
        '        + b"\\nendstream\\nendobj\\n"\n'
        '    )\n'
        '    xref_off = off3 + len(o3)\n'
        '    mini = (\n'
        '        hdr + o1 + o2 + o3\n'
        '        + f"xref\\n0 4\\n0000000000 65535 f \\n{off1:010d} 00000 n \\n{off2:010d} 00000 n \\n{off3:010d} 00000 n \\n".encode("ascii")\n'
        '        + f"trailer\\n<< /Size 4 /Root 1 0 R >>\\nstartxref\\n{xref_off}\\n%%EOF".encode("ascii")\n'
        '    )\n'
        '    d = fitz.open(stream=mini, filetype="pdf")\n'
        '    out = d.xref_stream(3)\n'
        '    d.close()\n'
        '    return out\n\n\n'
        'def _shannon_entropy(raw_bytes):\n'
        '    n = len(raw_bytes)\n'
        '    if n == 0:\n'
        '        return 0.0\n'
        '    arr = np.frombuffer(raw_bytes, dtype=np.uint8)\n'
        '    counts = np.bincount(arr, minlength=256)\n'
        '    probs = counts[counts > 0] / float(n)\n'
        '    return float(-np.sum(probs * np.log2(probs)))\n\n\n'
        'def _fast_sig_hash(data_bytes):\n'
        '    n = len(data_bytes)\n'
        '    if n == 0:\n'
        '        return "0:0000000000000000"\n'
        '    arr = np.frombuffer(data_bytes, dtype=np.uint8)\n'
        '    if n <= 8192:\n'
        '        sample = arr\n'
        '    else:\n'
        '        step = max(1, n // 4096)\n'
        '        sample = np.concatenate([arr[:2048], arr[::step][:4096], arr[-2048:]])\n'
        '    h = 14695981039346656037\n'
        '    for b in sample.tolist():\n'
        '        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF\n'
        '    return f"{n}:{h:016x}"\n\n\n'
        'def _split_delimited_line(line, sep):\n'
        '    if \'"\' not in line:\n'
        '        return line.split(sep)\n'
        '    out, cur, in_q = [], [], False\n'
        '    i, n = 0, len(line)\n'
        '    while i < n:\n'
        '        ch = line[i]\n'
        '        if ch == \'"\':\n'
        '            if in_q and i + 1 < n and line[i + 1] == \'"\':\n'
        '                cur.append(\'"\')\n'
        '                i += 1\n'
        '            else:\n'
        '                in_q = not in_q\n'
        '        elif ch == sep and not in_q:\n'
        '            out.append("".join(cur))\n'
        '            cur = []\n'
        '        else:\n'
        '            cur.append(ch)\n'
        '        i += 1\n'
        '    out.append("".join(cur))\n'
        '    return out\n\n\n'
        'def _profile_tabular_datasets(tabular_groups):\n'
        '    summaries = []\n'
        '    for ds_name, info in sorted(tabular_groups.items()):\n'
        '        cols = info["columns"]\n'
        '        sample_rows = info["sample_rows"]\n'
        '        col_stats = []\n'
        '        if cols and sample_rows:\n'
        '            for ci, cname in enumerate(cols[:25]):\n'
        '                vals = [r[ci].strip() for r in sample_rows if ci < len(r) and r[ci].strip() != ""]\n'
        '                nums = []\n'
        '                for v in vals[:600]:\n'
        '                    try:\n'
        '                        nums.append(float(v))\n'
        '                    except ValueError:\n'
        '                        pass\n'
        '                if len(nums) >= max(5, int(len(vals[:600]) * 0.7)):\n'
        '                    arr = np.array(nums, dtype=np.float64)\n'
        '                    col_stats.append({\n'
        '                        "col": cname,\n'
        '                        "type": "numeric",\n'
        '                        "min": round(float(arr.min()), 4),\n'
        '                        "max": round(float(arr.max()), 4),\n'
        '                        "mean": round(float(arr.mean()), 4),\n'
        '                    })\n'
        '                else:\n'
        '                    freq = {}\n'
        '                    for v in vals[:600]:\n'
        '                        freq[v] = freq.get(v, 0) + 1\n'
        '                    top_v = sorted(freq.items(), key=lambda x: -x[1])[:3]\n'
        '                    col_stats.append({\n'
        '                        "col": cname,\n'
        '                        "type": "text",\n'
        '                        "unique_in_sample": len(freq),\n'
        '                        "top_values": [f"{k} ({v})" for k, v in top_v],\n'
        '                    })\n'
        '        summaries.append({\n'
        '            "dataset": ds_name,\n'
        '            "chunks": info["chunks"],\n'
        '            "total_bytes": info["bytes"],\n'
        '            "total_data_rows": info["rows"],\n'
        '            "columns": cols[:30],\n'
        '            "column_profiles": col_stats,\n'
        '        })\n'
        '    return summaries\n\n\n'
        'def _load_parts_from_pdfs(pdf_paths=None):\n'
        '    targets = pdf_paths if pdf_paths else DEFAULT_PDFS\n'
        '    parts = []\n'
        '    for p in targets:\n'
        '        try:\n'
        '            doc = fitz.open(p)\n'
        '        except Exception:\n'
        '            continue\n'
        '        imgs = doc.get_page_images(0)\n'
        '        if not imgs:\n'
        '            doc.close()\n'
        '            continue\n'
        '        pix = fitz.Pixmap(doc, imgs[0][0])\n'
        '        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))[:, :, :3]\n'
        '        doc.close()\n'
        '        if arr.shape[0] <= 128:\n'
        '            continue\n'
        '        sub = arr[128:, :, :]\n'
        '        for sc in (1, 2, 4, 8):\n'
        '            if sub.shape[0] % sc or sub.shape[1] % sc:\n'
        '                continue\n'
        '            raw = sub.tobytes() if sc == 1 else sub[sc // 2 :: sc, sc // 2 :: sc, :].tobytes()\n'
        '            if raw[:4] == b"LLMP":\n'
        '                ver = raw[4]\n'
        '                flags = raw[5]\n'
        '                part_idx = raw[6]\n'
        '                clen = int(np.frombuffer(raw[8:12], dtype=">u4")[0])\n'
        '                payload = raw[12 : 12 + clen]\n'
        '                if ver == 2:\n'
        '                    stream = _inflate_via_fitz(payload) if (flags & 1) else payload\n'
        '                    toc_len = int(np.frombuffer(stream[:4], dtype=">u4")[0])\n'
        '                    toc = json.loads(stream[4 : 4 + toc_len].decode("utf-8"))\n'
        '                    blob = stream[4 + toc_len :]\n'
        '                    parts.append((part_idx, toc, blob))\n'
        '                break\n'
        '    parts.sort(key=lambda x: x[0])\n'
        '    return parts\n\n\n'
        'def run(preloaded=None, pdf_paths=None, top_n=25, max_stdout_chars=18000):\n'
        '    """\n'
        '    Generic archive profiler: deduplicates entries, measures Shannon entropy,\n'
        '    computes information weight, and summarizes tabular CSV/TSV datasets.\n'
        '    """\n'
        '    parts = preloaded if preloaded is not None else _load_parts_from_pdfs(pdf_paths)\n'
        '    if not parts:\n'
        '        msg = {"error": "No LLMPv2 PDFs found in working directory.", "checked": pdf_paths or DEFAULT_PDFS}\n'
        '        print(json.dumps(msg, indent=2))\n'
        '        return msg\n\n'
        '    seen = {}\n'
        '    dup_count = 0\n'
        '    ranked = []\n'
        '    tabular_groups = {}\n'
        '    total_entries = 0\n'
        '    total_raw_bytes = 0\n'
        '    entropy_vals = []\n\n'
        '    for part_idx, toc, blob in parts:\n'
        '        block_lookup = {}\n'
        '        for b in toc.get("blocks", []):\n'
        '            for g in range(b["global_file_start"], b["global_file_end"] + 1):\n'
        '                block_lookup[g] = b["block_id"]\n\n'
        '        for f in toc["files"]:\n'
        '            if f.get("virtual") or f["name"] == "00_scan.py":\n'
        '                continue\n'
        '            total_entries += 1\n'
        '            total_raw_bytes += f["len"]\n'
        '            raw_slice = blob[f["offset"] : f["offset"] + f["len"]]\n'
        '            parent_name = f.get("parent_file", f["name"])\n'
        '            low_pname = parent_name.lower()\n\n'
        '            if low_pname.endswith((".csv", ".tsv")):\n'
        '                sep = "\\t" if low_pname.endswith(".tsv") else ","\n'
        '                txt = raw_slice.decode("utf-8", "replace")\n'
        '                tlines = [ln for ln in txt.splitlines() if ln.strip()]\n'
        '                grp = tabular_groups.setdefault(\n'
        '                    parent_name,\n'
        '                    {"columns": [], "chunks": 0, "bytes": 0, "rows": 0, "sample_rows": []},\n'
        '                )\n'
        '                grp["chunks"] += 1\n'
        '                grp["bytes"] += f["len"]\n'
        '                if tlines:\n'
        '                    if not grp["columns"]:\n'
        '                        grp["columns"] = [c.strip() for c in _split_delimited_line(tlines[0], sep)]\n'
        '                    data_lines = tlines[1:]\n'
        '                    grp["rows"] += len(data_lines)\n'
        '                    if len(grp["sample_rows"]) < 600 and data_lines:\n'
        '                        step = max(1, len(data_lines) // 40)\n'
        '                        for rln in data_lines[::step][:40]:\n'
        '                            if len(grp["sample_rows"]) >= 600:\n'
        '                                break\n'
        '                            grp["sample_rows"].append(_split_delimited_line(rln, sep))\n'
        '                continue\n\n'
        '            txt = raw_slice.decode("utf-8", "replace")\n'
        '            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]\n'
        '            norm = " ".join(" ".join(lines).lower().split())\n'
        '            sig_hash = _fast_sig_hash(norm.encode("utf-8") if norm else raw_slice)\n'
        '            if sig_hash in seen:\n'
        '                dup_count += 1\n'
        '                seen[sig_hash]["dups"] += 1\n'
        '                continue\n\n'
        '            ent = round(_shannon_entropy(raw_slice), 3)\n'
        '            entropy_vals.append(ent)\n'
        '            uniq_ratio = (len(set(lines)) / len(lines)) if lines else 0.0\n'
        '            weight = round(ent * (0.25 + 0.75 * uniq_ratio) * float(np.log1p(len(raw_slice))), 2)\n\n'
        '            rec = {\n'
        '                "global_idx": f.get("global_idx", -1),\n'
        '                "block": block_lookup.get(f.get("global_idx", -1), f"PART_{part_idx}"),\n'
        '                "filename": f["name"],\n'
        '                "bytes": f["len"],\n'
        '                "entropy": ent,\n'
        '                "uniq_line_ratio": round(uniq_ratio, 3),\n'
        '                "weight": weight,\n'
        '                "dups": 0,\n'
        '                "preview": " ".join(lines)[:140],\n'
        '            }\n'
        '            seen[sig_hash] = rec\n'
        '            ranked.append(rec)\n\n'
        '    ranked.sort(key=lambda r: (-r["weight"], -r["entropy"], r["global_idx"]))\n'
        '    tabular_summaries = _profile_tabular_datasets(tabular_groups)\n'
        '    mean_ent = round(float(np.mean(entropy_vals)), 3) if entropy_vals else 0.0\n'
        '    report = {\n'
        '        "parts_scanned": [p[0] for p in parts],\n'
        '        "total_entries": total_entries,\n'
        '        "unique_text_entries": len(ranked),\n'
        '        "duplicates_collapsed": dup_count,\n'
        '        "total_raw_bytes": total_raw_bytes,\n'
        '        "mean_shannon_entropy": mean_ent,\n'
        '        "tabular_datasets": tabular_summaries,\n'
        '        "top_weighted_entries": ranked[:top_n],\n'
        '    }\n\n'
        '    try:\n'
        '        with open("scan_report.json", "w", encoding="utf-8") as out_f:\n'
        '            json.dump(report, out_f, indent=2)\n'
        '    except Exception:\n'
        '        pass\n\n'
        '    trimmed = dict(report)\n'
        '    while len(json.dumps(trimmed, indent=2)) > max_stdout_chars and len(trimmed["top_weighted_entries"]) > 5:\n'
        '        trimmed["top_weighted_entries"] = trimmed["top_weighted_entries"][:-1]\n\n'
        '    print(json.dumps(trimmed, indent=2))\n'
        '    return report\n\n\n'
        'run_scan = run\n\n\n'
        'if __name__ == "__main__":\n'
        '    run()\n'
    )


class Progress:
    """Throttled single-line terminal status reporter for Termux."""

    def __init__(self):
        self.is_tty = sys.stdout.isatty()
        self.last_t = 0.0
        self.last_len = 0

    def update(self, msg: str, force: bool = False):
        now = time.monotonic()
        if not force and (now - self.last_t) < 0.12:
            return
        self.last_t = now
        cols = shutil.get_terminal_size((72, 24)).columns
        clipped = msg[: max(20, cols - 1)]
        if self.is_tty:
            pad = " " * max(0, self.last_len - len(clipped))
            sys.stdout.write(f"\r{clipped}{pad}")
            sys.stdout.flush()
            self.last_len = len(clipped)
        elif force:
            print(clipped, flush=True)

    def done(self, msg: str):
        if self.is_tty:
            pad = " " * max(0, self.last_len - len(msg))
            sys.stdout.write(f"\r{msg}{pad}\n")
            sys.stdout.flush()
            self.last_len = 0
        else:
            print(msg, flush=True)


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1_048_576:
        return f"{n / 1024:.1f}KB"
    return f"{n / 1_048_576:.2f}MB"


def max_frame_capacity(max_edge: int, scale: int) -> int:
    """Exact maximum frame byte capacity inside max_edge x max_edge below HEADER_H."""
    grid_w = max_edge // scale
    grid_h = max(1, (max_edge - HEADER_H) // scale)
    return grid_w * grid_h * 3


def load_mono_font(size: int):
    for path in (
        "/system/fonts/DroidSansMono.ttf",
        "/system/fonts/RobotoMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def find_subfolders(src_dir: str) -> list[str]:
    subs = []
    try:
        for entry in sorted(os.listdir(src_dir)):
            if entry.startswith(".") or entry in SKIP_DIRS:
                continue
            if os.path.isdir(os.path.join(src_dir, entry)):
                subs.append(entry)
    except OSError:
        pass
    return subs


def iter_entry_bytes(entry: dict):
    """Yields raw bytes for a normal file or a virtual row-sliced file chunk in 64 KB buffers."""
    prepend = entry.get("prepend_header", b"")
    if prepend:
        yield prepend
    start = entry.get("byte_start", 0)
    end = entry.get("byte_end", entry["len"])
    rem = end - start
    if rem <= 0:
        return
    with open(entry["full_path"], "rb") as f:
        if start > 0:
            f.seek(start)
        while rem > 0:
            buf = f.read(min(STREAM_CHUNK, rem))
            if not buf:
                break
            rem -= len(buf)
            yield buf


def extract_outline_from_head(name: str, head_bytes: bytes) -> list[str]:
    base = name.split(" [chunk ", 1)[0]
    ext = os.path.splitext(base)[1].lower()
    text = head_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if ext == ".json":
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return [f"JSON Array[{len(obj)}]"]
            if isinstance(obj, dict):
                return [f"key:{k}" for k in list(obj.keys())[:10]]
        except Exception:
            keys = re.findall(r'"([^"\\]{1,32})"\s*:', text)
            if keys:
                return [f"key:{k}" for k in list(dict.fromkeys(keys))[:8]]
    if ext in {".md", ".markdown"}:
        heads = [l.strip()[:48] for l in lines if re.match(r"^#{1,3}\s+", l)][:8]
        if heads:
            return heads
    if ext in {".py", ".js", ".ts"}:
        defs = [
            l.strip()[:48]
            for l in lines
            if l.strip().startswith(("def ", "class ", "function ", "export function "))
        ][:8]
        if defs:
            return defs
    if ext in TABULAR_EXTS and lines:
        return [f"cols: {lines[0][:60]}"]
    return [l.strip()[:48] for l in lines if l.strip()][:2]


def slice_large_file(
    rel: str,
    full: str,
    file_len: int,
    max_block_bytes: int,
    start_gidx: int,
    prog: Progress,
) -> list[dict]:
    """
    Splits oversized line-oriented files (CSV, TSV, JSONL, NDJSON, LOG) along newline
    boundaries into <= max_block_bytes virtual slices in O(64 KB) RAM.
    For CSV/TSV files, prepends the column header row to slices 2..N so every
    INDEX_XX block is a valid standalone table.
    """
    ext = os.path.splitext(rel)[1].lower()
    is_tabular = ext in TABULAR_EXTS
    header_bytes = b""

    with open(full, "rb") as f:
        first_buf = f.read(min(STREAM_CHUNK, file_len))
        if b"\x00" in first_buf[:4096]:
            return []
        if is_tabular:
            nl_pos = first_buf.find(b"\n")
            if 0 < nl_pos <= 16_384:
                header_bytes = first_buf[: nl_pos + 1]

        f.seek(0)
        splits: list[tuple[int, int]] = []
        chunk_start = 0
        pos = 0
        last_nl = -1

        while True:
            buf = f.read(STREAM_CHUNK)
            if not buf:
                break
            search_from = 0
            while True:
                idx = buf.find(b"\n", search_from)
                if idx == -1:
                    break
                abs_nl = pos + idx + 1
                last_nl = abs_nl
                if (abs_nl - chunk_start) >= max_block_bytes:
                    splits.append((chunk_start, abs_nl))
                    chunk_start = abs_nl
                    last_nl = -1
                search_from = idx + 1

            pos += len(buf)
            if (pos - chunk_start) >= int(max_block_bytes * 1.5):
                cut = last_nl if last_nl > chunk_start else pos
                splits.append((chunk_start, cut))
                chunk_start = cut
                last_nl = -1

        if chunk_start < file_len:
            splits.append((chunk_start, file_len))

    total_chunks = len(splits)
    out_entries: list[dict] = []
    for ci, (b_start, b_end) in enumerate(splits, 1):
        prog.update(f"[1/4] Row-slicing {os.path.basename(rel)} ({ci}/{total_chunks})...")
        prepend = header_bytes if (is_tabular and ci > 1 and header_bytes) else b""
        h = hashlib.sha256()
        slice_len = 0
        head = b""
        temp_e = {
            "full_path": full,
            "byte_start": b_start,
            "byte_end": b_end,
            "len": b_end - b_start,
            "prepend_header": prepend,
        }
        for piece in iter_entry_bytes(temp_e):
            if slice_len < OUTLINE_PEEK_BYTES:
                head += piece[: OUTLINE_PEEK_BYTES - slice_len]
            slice_len += len(piece)
            h.update(piece)

        chunk_name = rel if total_chunks == 1 else f"{rel} [chunk {ci:02d}/{total_chunks:02d}]"
        entry = {
            "global_idx": start_gidx + ci - 1,
            "name": chunk_name,
            "full_path": full,
            "byte_start": b_start,
            "byte_end": b_end,
            "prepend_header": prepend,
            "len": slice_len,
            "sha256": h.hexdigest(),
            "head": head,
        }
        if total_chunks > 1:
            entry["parent_file"] = rel
            entry["chunk_index"] = ci
            entry["chunk_total"] = total_chunks
        out_entries.append(entry)

    return out_entries


def scan_files_streaming(
    src_dir: str,
    recursive: bool,
    exts: set[str],
    self_name: str,
    max_raw_bytes: int,
    max_block_bytes: int,
    skip_scanner_name: bool,
    prog: Progress,
) -> tuple[list[dict], int]:
    """
    Pass 1: Walks directory, skips binaries, computes exact byte size and SHA-256,
    and automatically slices oversized CSV/TSV/JSONL/LOG files into <= max_block_bytes
    virtual chunks without loading entire files into RAM.
    """
    entries: list[dict] = []
    total_raw = 0
    scanned_count = 0

    def iter_candidates():
        if recursive:
            for root, dirs, filenames in os.walk(src_dir):
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d not in SKIP_DIRS]
                for fn in sorted(filenames):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, src_dir).replace("\\", "/")
                    yield rel, full
        else:
            for fn in sorted(os.listdir(src_dir)):
                full = os.path.join(src_dir, fn)
                if os.path.isfile(full):
                    yield fn, full

    for rel, full in iter_candidates():
        scanned_count += 1
        base = os.path.basename(rel)
        if base == self_name or base.startswith("llmp_"):
            continue
        if skip_scanner_name and rel == SCANNER_FILENAME:
            continue
        ext = os.path.splitext(base)[1].lower()
        if exts and ext not in exts:
            continue

        prog.update(f"[1/4] Scanning ({len(entries)} entries / {scanned_count} seen, {fmt_bytes(total_raw)}): {base}")

        try:
            disk_size = os.path.getsize(full)
            if disk_size == 0:
                continue
            if total_raw + disk_size > max_raw_bytes:
                raise MemoryError(
                    f"Input exceeds --max-raw cap ({fmt_bytes(max_raw_bytes)}). "
                    f"Use --ext to filter or raise --max-raw."
                )

            if disk_size > max_block_bytes and ext in CHUNKABLE_EXTS:
                sliced = slice_large_file(rel, full, disk_size, max_block_bytes, len(entries), prog)
                for se in sliced:
                    if total_raw + se["len"] > max_raw_bytes:
                        raise MemoryError(
                            f"Input exceeds --max-raw cap ({fmt_bytes(max_raw_bytes)}). "
                            f"Use --ext to filter or raise --max-raw."
                        )
                    entries.append(se)
                    total_raw += se["len"]
                continue

            h = hashlib.sha256()
            file_len = 0
            head = b""
            is_binary = False
            with open(full, "rb") as f:
                while True:
                    buf = f.read(STREAM_CHUNK)
                    if not buf:
                        break
                    if file_len == 0:
                        if b"\x00" in buf[:4096]:
                            is_binary = True
                            break
                        head = buf[:OUTLINE_PEEK_BYTES]
                    file_len += len(buf)
                    if total_raw + file_len > max_raw_bytes:
                        raise MemoryError(
                            f"Input exceeds --max-raw cap ({fmt_bytes(max_raw_bytes)}). "
                            f"Use --ext to filter or raise --max-raw."
                        )
                    h.update(buf)

            if is_binary or file_len == 0:
                continue

            entries.append({
                "global_idx": len(entries),
                "name": rel,
                "full_path": full,
                "byte_start": 0,
                "byte_end": file_len,
                "prepend_header": b"",
                "len": file_len,
                "sha256": h.hexdigest(),
                "head": head,
            })
            total_raw += file_len
        except OSError:
            continue

    keep_outlines = len(entries) <= COMPACT_TOC_THRESHOLD
    for e in entries:
        e["outline"] = extract_outline_from_head(os.path.basename(e["name"]), e["head"]) if keep_outlines else []
        del e["head"]

    prog.done(f"[1/4] Scan complete: {len(entries)} entry/entries | {fmt_bytes(total_raw)} raw")
    return entries, total_raw


def build_logical_blocks(entries: list[dict], max_block_bytes: int) -> list[dict]:
    """
    Groups sorted entries into contiguous logical blocks of <= max_block_bytes raw
    (default 500 KB = ~125K tokens) so an LLM can inspect one block at a time.
    """
    blocks: list[dict] = []
    if not entries:
        return blocks

    cur_start = 0
    cur_bytes = 0
    for idx, e in enumerate(entries):
        flen = e["len"]
        if cur_bytes > 0 and (cur_bytes + flen) > max_block_bytes:
            end_idx = idx - 1
            bnum = len(blocks) + 1
            blocks.append({
                "block_num": bnum,
                "block_id": f"INDEX_{bnum:02d}",
                "file_start": cur_start,
                "file_end": end_idx,
                "file_count": end_idx - cur_start + 1,
                "raw_bytes": cur_bytes,
                "first_file": entries[cur_start]["name"],
                "last_file": entries[end_idx]["name"],
            })
            cur_start = idx
            cur_bytes = 0
        cur_bytes += flen

    end_idx = len(entries) - 1
    bnum = len(blocks) + 1
    blocks.append({
        "block_num": bnum,
        "block_id": f"INDEX_{bnum:02d}",
        "file_start": cur_start,
        "file_end": end_idx,
        "file_count": end_idx - cur_start + 1,
        "raw_bytes": cur_bytes,
        "first_file": entries[cur_start]["name"],
        "last_file": entries[end_idx]["name"],
    })
    return blocks


def build_directory_tree(entries: list[dict], max_lines: int = 40) -> list[str]:
    """Summarizes folder hierarchy with entry counts and raw byte sizes."""
    dir_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in entries:
        raw_name = e.get("parent_file", e["name"])
        d = os.path.dirname(raw_name) or "."
        dir_stats[d][0] += 1
        dir_stats[d][1] += e["len"]

    sorted_dirs = sorted(dir_stats.items(), key=lambda kv: kv[0])
    lines = []
    for d, (cnt, sz) in sorted_dirs[:max_lines]:
        label = "./" if d == "." else f"{d}/"
        lines.append(f"  {label:<36} | {cnt:5d} entry(s) | {fmt_bytes(sz):>8} (~{math.ceil(sz / 4):,} tok)")
    if len(sorted_dirs) > max_lines:
        rem_dirs = len(sorted_dirs) - max_lines
        rem_files = sum(v[0] for _, v in sorted_dirs[max_lines:])
        rem_bytes = sum(v[1] for _, v in sorted_dirs[max_lines:])
        lines.append(f"  ... (+{rem_dirs} more dirs, {rem_files} entries, {fmt_bytes(rem_bytes)})")
    return lines


def plan_self_contained_parts(
    entries: list[dict],
    blocks: list[dict],
    compressed: bool,
    max_payload_bytes: int,
    embed_scan: bool,
    scanner_code: str,
    prog: Progress,
) -> list[list[dict]]:
    """
    Pass 2: Measures each logical block's exact TOC + file payload footprint
    using constant-memory streaming compression, then bin-packs whole blocks
    into self-contained parts <= max_payload_bytes.
    """
    part_groups: list[list[dict]] = []
    cur_blocks: list[dict] = []
    scanner_bytes = scanner_code.encode("utf-8") if embed_scan else b""
    scanner_overhead = (len(zlib.compress(scanner_bytes, level=6)) + 160) if embed_scan else 0
    cur_packed_est = scanner_overhead
    safe_ceiling = max(32_768, max_payload_bytes - 8_192)

    for bi, b in enumerate(blocks, 1):
        prog.update(f"[2/4] Sizing block {bi}/{len(blocks)} ({b['block_id']}) for standalone parts...")

        sample_files = []
        for fi in range(b["file_start"], b["file_end"] + 1):
            e = entries[fi]
            item = {
                "global_idx": e["global_idx"],
                "name": e["name"],
                "offset": 99_999_999,
                "len": e["len"],
                "sha256": e["sha256"],
            }
            if "parent_file" in e:
                item["parent_file"] = e["parent_file"]
                item["chunk_index"] = e["chunk_index"]
                item["chunk_total"] = e["chunk_total"]
            if e["outline"]:
                item["outline"] = e["outline"]
            sample_files.append(item)

        sample_block = {
            "block_num": b["block_num"],
            "block_id": b["block_id"],
            "global_file_start": b["file_start"],
            "global_file_end": b["file_end"],
            "local_file_start": 9999,
            "local_file_end": 9999,
            "file_count": b["file_count"],
            "raw_bytes": b["raw_bytes"],
            "first_file": b["first_file"],
            "last_file": b["last_file"],
        }
        toc_sample_bytes = (
            json.dumps(sample_block, separators=(",", ":")).encode("utf-8")
            + json.dumps(sample_files, separators=(",", ":")).encode("utf-8")
        )
        del sample_files

        if compressed:
            cobj = zlib.compressobj(level=6)
            needed = len(cobj.compress(toc_sample_bytes))
            for fi in range(b["file_start"], b["file_end"] + 1):
                for buf in iter_entry_bytes(entries[fi]):
                    needed += len(cobj.compress(buf))
            needed += len(cobj.flush(zlib.Z_FINISH))
        else:
            needed = len(toc_sample_bytes) + b["raw_bytes"]

        if needed + scanner_overhead > safe_ceiling:
            raise MemoryError(
                f"Block {b['block_id']} ({fmt_bytes(needed)} packed) exceeds single-part capacity "
                f"({fmt_bytes(safe_ceiling)}). Increase --max-edge or lower --block-kb."
            )

        if cur_blocks and (cur_packed_est + needed) > safe_ceiling:
            part_groups.append(cur_blocks)
            cur_blocks = []
            cur_packed_est = scanner_overhead

        cur_blocks.append(b)
        cur_packed_est += needed

    if cur_blocks:
        part_groups.append(cur_blocks)

    for p_idx, grp in enumerate(part_groups, 1):
        for b in grp:
            b["part_num"] = p_idx
            b["part_total"] = len(part_groups)

    prog.done(
        f"[2/4] Planned {len(part_groups)} self-contained part(s) across {len(blocks)} logical block(s)"
    )
    return part_groups


def build_part_payload(
    entries: list[dict],
    part_blocks: list[dict],
    part_idx: int,
    part_total: int,
    compressed: bool,
    embed_scan: bool,
    scanner_code: str,
) -> tuple[bytes, int, int]:
    """
    Builds one standalone binary payload: [4B toc_len][part_toc_json][part_files_blob],
    optionally injecting virtual 00_scan.py at local_idx 0 without shifting user indices.
    """
    local_offset = 0
    local_files = []
    local_blocks = []
    scanner_bytes = b""

    if embed_scan:
        scanner_bytes = scanner_code.encode("utf-8")
        local_files.append({
            "global_idx": -1,
            "virtual": True,
            "name": SCANNER_FILENAME,
            "offset": 0,
            "len": len(scanner_bytes),
            "sha256": hashlib.sha256(scanner_bytes).hexdigest(),
        })
        local_offset += len(scanner_bytes)

    user_file_count = 0
    for b in part_blocks:
        l_start = len(local_files)
        for fi in range(b["file_start"], b["file_end"] + 1):
            e = entries[fi]
            item = {
                "global_idx": e["global_idx"],
                "name": e["name"],
                "offset": local_offset,
                "len": e["len"],
                "sha256": e["sha256"],
            }
            if "parent_file" in e:
                item["parent_file"] = e["parent_file"]
                item["chunk_index"] = e["chunk_index"]
                item["chunk_total"] = e["chunk_total"]
            if e["outline"]:
                item["outline"] = e["outline"]
            local_files.append(item)
            local_offset += e["len"]
            user_file_count += 1
        l_end = len(local_files) - 1

        local_blocks.append({
            "block_num": b["block_num"],
            "block_id": b["block_id"],
            "global_file_start": b["file_start"],
            "global_file_end": b["file_end"],
            "local_file_start": l_start,
            "local_file_end": l_end,
            "file_count": b["file_count"],
            "raw_bytes": b["raw_bytes"],
            "first_file": b["first_file"],
            "last_file": b["last_file"],
        })

    toc_obj = {
        "schema": "LLMPv2",
        "part": part_idx,
        "total_parts": part_total,
        "embedded_scanner": SCANNER_FILENAME if embed_scan else None,
        "blocks": local_blocks,
        "files": local_files,
    }
    toc_bytes = json.dumps(toc_obj, separators=(",", ":")).encode("utf-8")
    toc_hdr = struct.pack(">I", len(toc_bytes)) + toc_bytes

    if compressed:
        cobj = zlib.compressobj(level=6)
        out = bytearray(cobj.compress(toc_hdr))
        if embed_scan:
            out.extend(cobj.compress(scanner_bytes))
        for b in part_blocks:
            for fi in range(b["file_start"], b["file_end"] + 1):
                for buf in iter_entry_bytes(entries[fi]):
                    out.extend(cobj.compress(buf))
        out.extend(cobj.flush(zlib.Z_FINISH))
        payload = bytes(out)
    else:
        out = bytearray(toc_hdr)
        if embed_scan:
            out.extend(scanner_bytes)
        for b in part_blocks:
            for fi in range(b["file_start"], b["file_end"] + 1):
                for buf in iter_entry_bytes(entries[fi]):
                    out.extend(buf)
        payload = bytes(out)

    return payload, local_offset, user_file_count


def solve_dimensions(frame_len: int, max_edge: int, scale: int = DEFAULT_SCALE) -> dict:
    """
    Solves optimal image dimensions <= max_edge x max_edge for the target pixel scale
    (default 1x1), optimizing proximity to a 1.6:1 aspect ratio without overflow.
    """
    logical_px = math.ceil(frame_len / 3)
    candidates = [w for w in (512, 768, 1024, 1280, 1536, 1792, 2048) if w <= max_edge]
    if max_edge not in candidates:
        candidates.append(max_edge)
    candidates = sorted(set(candidates))

    best = None
    min_grid_h = max(1, math.ceil(32 / scale))

    for img_w in candidates:
        grid_w = img_w // scale
        if grid_w <= 0:
            continue
        grid_h = max(min_grid_h, math.ceil(logical_px / grid_w))
        img_h = HEADER_H + grid_h * scale
        if img_h <= max_edge:
            ratio = img_w / img_h
            score = abs(math.log(ratio / 1.6))
            if best is None or score < best["score"]:
                best = {
                    "img_w": img_w,
                    "img_h": img_h,
                    "grid_w": grid_w,
                    "grid_h": grid_h,
                    "scale": scale,
                    "score": score,
                }

    if best is None:
        img_w = candidates[-1]
        grid_w = img_w // scale
        grid_h = math.ceil(logical_px / grid_w)
        best = {
            "img_w": img_w,
            "img_h": HEADER_H + grid_h * scale,
            "grid_w": grid_w,
            "grid_h": grid_h,
            "scale": scale,
        }
    return best


def render_part_rgb(frame_bytes: bytes, part_idx: int, part_total: int, meta: dict) -> tuple[np.ndarray, str, dict, bool]:
    scale = meta["scale"]
    dim = solve_dimensions(len(frame_bytes), meta["max_edge"], scale=scale)
    img_w, img_h, grid_w, grid_h = dim["img_w"], dim["img_h"], dim["grid_w"], dim["grid_h"]

    if img_h > meta["max_edge"] or img_w > meta["max_edge"]:
        raise RuntimeError(
            f"Solved dimensions {img_w}x{img_h} exceed --max-edge {meta['max_edge']}"
        )

    sha = hashlib.sha256(frame_bytes).hexdigest()

    hdr_img = Image.new("RGB", (img_w, HEADER_H), (255, 212, 0))
    draw = ImageDraw.Draw(hdr_img)

    cal_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
    cal_x = img_w - 114
    for i, col in enumerate(cal_colors):
        x0 = cal_x + i * 21
        outline = (0, 0, 0) if col == (255, 255, 255) else None
        draw.rectangle([x0, 8, x0 + 17, 25], fill=col, outline=outline)

    f_bold = load_mono_font(13)
    f_reg = load_mono_font(12)
    comp_label = "zlib" if meta["compressed"] else "raw"
    mode_tag = "1x1 NATIVE" if scale == 1 else f"{scale}x MACRO"
    scan_tag = f" | TOOL={SCANNER_FILENAME}" if meta.get("embed_scan") else ""

    draw.text((10, 8), f"LLMPv2 ({mode_tag}) | PART={part_idx}/{part_total} | COMPRESS={comp_label} | SCALE={scale}x | GRID={grid_w}x{grid_h}{scan_tag}", fill=(0, 0, 0), font=f_bold)
    draw.text((10, 28), f"PART_RAW={meta['part_raw_len']} | FRAME_LEN={len(frame_bytes)} | PART_FILES={meta['part_file_count']} | BLOCKS={meta['block_span']} | IMG={img_w}x{img_h}", fill=(0, 0, 0), font=f_reg)
    draw.text((10, 48), f"SHA256={sha}", fill=(0, 0, 0), font=f_reg)
    if scale == 1:
        dec_line = f"DECODE (fitz+numpy): Pixmap RGB[y>={HEADER_H}], bytes[:FRAME_LEN], skip 12B hdr{', inflate via PDF FlateDecode' if meta['compressed'] else ''}, parse TOC"
    else:
        dec_line = f"DECODE ({scale}x TILE): Pixmap RGB[y>={HEADER_H}], sample {scale}x{scale} center px, skip 12B hdr{', inflate via PDF FlateDecode' if meta['compressed'] else ''}, parse TOC"
    draw.text((10, 68), dec_line, fill=(0, 0, 0), font=f_reg)
    max_chars = max(20, int((img_w - 20) / 7.2))
    draw.text((10, 88), f"CONTENTS: {meta['file_preview']}"[:max_chars], fill=(0, 0, 0), font=f_reg)

    total_bytes = grid_w * grid_h * 3
    padded = np.zeros(total_bytes, dtype=np.uint8)
    padded[:len(frame_bytes)] = np.frombuffer(frame_bytes, dtype=np.uint8)
    logical_grid = padded.reshape((grid_h, grid_w, 3))

    if scale == 1:
        data_arr = logical_grid
    else:
        data_arr = np.repeat(np.repeat(logical_grid, scale, axis=0), scale, axis=1)

    full_arr = np.vstack([np.array(hdr_img, dtype=np.uint8), data_arr])
    del padded, logical_grid, data_arr

    sub = full_arr[HEADER_H:, :, :]
    if scale == 1:
        recovered = sub.tobytes()[:len(frame_bytes)]
    else:
        half = scale // 2
        recovered = sub[half::scale, half::scale, :].tobytes()[:len(frame_bytes)]

    verified = hashlib.sha256(recovered).hexdigest() == sha
    del recovered

    return full_arr, sha, dim, verified


def write_lossless_pdf(pdf_path: str, rgb_arr: np.ndarray):
    """Writes a spec-compliant PDF 1.4 /FlateDecode Image XObject directly to disk."""
    h, w, c = rgb_arr.shape
    assert c == 3 and rgb_arr.dtype == np.uint8
    comp_rgb = zlib.compress(rgb_arr.tobytes(), level=3)

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
        f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    ).encode("ascii")
    obj4_hdr = (
        f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(comp_rgb)} >>\nstream\n"
    ).encode("ascii")
    obj4_ftr = b"\nendstream\nendobj\n"
    cstream = f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q\n".encode("ascii")
    obj5 = (
        f"5 0 obj\n<< /Length {len(cstream)} >>\nstream\n".encode("ascii")
        + cstream
        + b"endstream\nendobj\n"
    )

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = []
    pos = len(header)
    for sz in (
        len(obj1),
        len(obj2),
        len(obj3),
        len(obj4_hdr) + len(comp_rgb) + len(obj4_ftr),
        len(obj5),
    ):
        offsets.append(pos)
        pos += sz

    with open(pdf_path, "wb") as pf:
        pf.write(header)
        pf.write(obj1)
        pf.write(obj2)
        pf.write(obj3)
        pf.write(obj4_hdr)
        pf.write(comp_rgb)
        pf.write(obj4_ftr)
        pf.write(obj5)
        pf.write(f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("ascii"))
        for off in offsets:
            pf.write(f"{off:010d} 00000 n \n".encode("ascii"))
        pf.write(
            f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n".encode("ascii")
        )


def build_manifest(
    stamp: str,
    entries: list[dict],
    blocks: list[dict],
    tree_lines: list[str],
    total_raw: int,
    total_packed: int,
    compressed: bool,
    parts_info: list[dict],
    block_kb: int,
    scale: int,
    max_edge: int,
    embed_scan: bool,
) -> str:
    bt = "`" * 3
    cap_bytes = max_frame_capacity(max_edge, scale)
    ratio_pct = (100.0 * total_packed / total_raw) if total_raw else 0.0
    pdf_names = [p["pdf"] for p in parts_info]

    part_lines = [
        f"  Part {idx+1}/{len(parts_info)} (STANDALONE): {p['pdf']} / {p['png']}\n"
        f"    Blocks: {p['block_span']} | File Indices: [{p['file_span']}] ({p['part_files']:,} entries) | "
        f"Raw: {fmt_bytes(p['part_raw'])} (~{math.ceil(p['part_raw'] / 4):,} tok)\n"
        f"    Raster: {p['dim']['img_w']}x{p['dim']['img_h']} px | Scale: {p['dim']['scale']}x | "
        f"FRAME_LEN: {p['frame_len']:,} B | SHA256: {p['sha']}"
        for idx, p in enumerate(parts_info)
    ]

    block_lines = []
    for b in blocks:
        avg_sz = b["raw_bytes"] // max(1, b["file_count"])
        span = (
            b["first_file"]
            if b["file_start"] == b["file_end"]
            else f"{b['first_file']} -> {b['last_file']}"
        )
        block_lines.append(
            f"  {b['block_id']} | Part {b['part_num']}/{b['part_total']} | "
            f"entries [{b['file_start']}..{b['file_end']}] ({b['file_count']:3d} entries, avg {fmt_bytes(avg_sz):>6}) | "
            f"{fmt_bytes(b['raw_bytes']):>7} (~{math.ceil(b['raw_bytes'] / 4):>6,} tok) | {span}"
        )

    decoder_snippet = f"""import fitz, json
import numpy as np

PDF_FILES = {json.dumps(pdf_names)}

def inflate_via_fitz(comp_bytes):
    # Decompresses raw zlib payload in C via MuPDF FlateDecode (bypasses blocked `import zlib`)
    hdr = b"%PDF-1.4\\n"
    o1 = b"1 0 obj\\n<< /Type /Catalog /Pages 2 0 R >>\\nendobj\\n"
    o2 = b"2 0 obj\\n<< /Type /Pages /Kids [] /Count 0 >>\\nendobj\\n"
    off1 = len(hdr)
    off2 = off1 + len(o1)
    off3 = off2 + len(o2)
    o3 = (
        f"3 0 obj\\n<< /Length {{len(comp_bytes)}} /Filter /FlateDecode >>\\nstream\\n".encode("ascii")
        + comp_bytes
        + b"\\nendstream\\nendobj\\n"
    )
    xref_off = off3 + len(o3)
    mini = (
        hdr + o1 + o2 + o3
        + f"xref\\n0 4\\n0000000000 65535 f \\n{{off1:010d}} 00000 n \\n{{off2:010d}} 00000 n \\n{{off3:010d}} 00000 n \\n".encode("ascii")
        + f"trailer\\n<< /Size 4 /Root 1 0 R >>\\nstartxref\\n{{xref_off}}\\n%%EOF".encode("ascii")
    )
    d = fitz.open(stream=mini, filetype="pdf")
    out = d.xref_stream(3)
    d.close()
    return out

loaded_parts = []
blocks_by_num = {{}}
files_by_idx = {{}}
scanner_src = None
total_expected = 1

for p in PDF_FILES:
    try:
        doc = fitz.open(p)
    except Exception:
        continue
    imgs = doc.get_page_images(0)
    if not imgs:
        doc.close()
        continue
    pix = fitz.Pixmap(doc, imgs[0][0])
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))[:, :, :3]
    doc.close()
    if arr.shape[0] <= 128:
        continue
    sub = arr[128:, :, :]
    for sc in (1, 2, 4, 8):
        if sub.shape[0] % sc or sub.shape[1] % sc:
            continue
        raw = sub.tobytes() if sc == 1 else sub[sc // 2 :: sc, sc // 2 :: sc, :].tobytes()
        if raw[:4] == b"LLMP":
            ver, flags, part_idx, total_expected = raw[4], raw[5], raw[6], raw[7]
            clen = int(np.frombuffer(raw[8:12], dtype=">u4")[0])
            payload = raw[12 : 12 + clen]
            if ver == 2:
                stream = inflate_via_fitz(payload) if (flags & 1) else payload
                toc_len = int(np.frombuffer(stream[:4], dtype=">u4")[0])
                toc = json.loads(stream[4 : 4 + toc_len].decode("utf-8"))
                blob = stream[4 + toc_len :]
                loaded_parts.append((part_idx, toc, blob))
                p_files = toc["files"]
                for f in p_files:
                    if f.get("virtual") and f["name"] == "00_scan.py":
                        scanner_src = blob[f["offset"] : f["offset"] + f["len"]].decode("utf-8", "replace")
                        continue
                    gidx = f.get("global_idx", len(files_by_idx))
                    files_by_idx[gidx] = (f, blob)
                for b in toc.get("blocks", []):
                    blocks_by_num[b["block_num"]] = (b, p_files, blob)
            break

def block_summary():
    return [
        f"{{b['block_id']}} | entries [{{b['global_file_start']}}..{{b['global_file_end']}}] "
        f"({{b['file_count']}} entries, {{b['raw_bytes']}} B) | {{b['first_file']}} -> {{b['last_file']}}"
        for _, (b, _, _) in sorted(blocks_by_num.items())
    ]

def list_block(num):
    if num not in blocks_by_num:
        return f"Block INDEX_{{num:02d}} not in attached part(s). Loaded: {{sorted(blocks_by_num)}}"
    b, p_files, _ = blocks_by_num[num]
    return [
        f"[{{p_files[i]['global_idx']}}] {{p_files[i]['name']}} ({{p_files[i]['len']}} B)"
        for i in range(b["local_file_start"], b["local_file_end"] + 1)
    ]

def get_block(num, page=1, page_chars=18000):
    if num not in blocks_by_num:
        return f"Block INDEX_{{num:02d}} not in attached part(s). Loaded: {{sorted(blocks_by_num)}}"
    b, p_files, blob = blocks_by_num[num]
    out = []
    for f in p_files[b["local_file_start"] : b["local_file_end"] + 1]:
        txt = blob[f["offset"] : f["offset"] + f["len"]].decode("utf-8", "replace")
        out.append(f"=== [{{f['global_idx']}}] {{f['name']}} ({{f['len']}} B) ===\\n{{txt}}")
    joined = "\\n".join(out)
    start = (page - 1) * page_chars
    pages = max(1, (len(joined) + page_chars - 1) // page_chars)
    return f"[{{b['block_id']}} | entries {{b['global_file_start']}}..{{b['global_file_end']}} | page {{page}}/{{pages}}]\\n" + joined[start : start + page_chars]

def read_file(gidx, page=1, page_chars=18000):
    if gidx not in files_by_idx:
        return f"Entry index {{gidx}} not in attached part(s)."
    f, blob = files_by_idx[gidx]
    txt = blob[f["offset"] : f["offset"] + f["len"]].decode("utf-8", "replace")
    start = (page - 1) * page_chars
    pages = max(1, (len(txt) + page_chars - 1) // page_chars)
    return f"=== [{{gidx}}] {{f['name']}} ({{f['len']}} B | page {{page}}/{{pages}}) ===\\n" + txt[start : start + page_chars]

def search_files(query, max_hits=25):
    q = query.lower()
    hits = []
    for gidx in sorted(files_by_idx):
        f, blob = files_by_idx[gidx]
        txt = blob[f["offset"] : f["offset"] + f["len"]].decode("utf-8", "replace")
        low = txt.lower()
        pos = 0
        while len(hits) < max_hits:
            pos = low.find(q, pos)
            if pos == -1:
                break
            line_num = txt.count("\\n", 0, pos) + 1
            s = max(0, pos - 60)
            snippet = txt[s : pos + len(q) + 100].replace("\\n", " ")
            hits.append(f"[{{gidx}} | L{{line_num}}] {{f['name']}}: ...{{snippet}}...")
            pos += len(q)
        if len(hits) >= max_hits:
            break
    return "\\n".join(hits) if hits else f"No matches for '{{query}}'."

def query_csv(query="", col=None, max_hits=25, file_filter=""):
    q = query.lower()
    ff = file_filter.lower()
    hits = []
    total_scanned = 0
    for gidx in sorted(files_by_idx):
        f, blob = files_by_idx[gidx]
        pname = f.get("parent_file", f["name"])
        if ff and ff not in pname.lower():
            continue
        if not pname.lower().endswith((".csv", ".tsv")):
            continue
        sep = "\\t" if pname.lower().endswith(".tsv") else ","
        txt = blob[f["offset"] : f["offset"] + f["len"]].decode("utf-8", "replace")
        lines = txt.splitlines()
        if not lines:
            continue
        headers = [h.strip() for h in lines[0].split(sep)]
        col_idx = None
        if col is not None:
            for ci, h in enumerate(headers):
                if h.lower() == col.lower():
                    col_idx = ci
                    break
            if col_idx is None:
                return f"Column '{{col}}' not found. Available: {{headers}}"
        for r_off, row in enumerate(lines[1:], 1):
            if not row.strip():
                continue
            total_scanned += 1
            cells = row.split(sep)
            target_val = cells[col_idx] if (col_idx is not None and col_idx < len(cells)) else row
            if not q or q in target_val.lower():
                if len(hits) < max_hits:
                    hits.append(f"[{{gidx}} | row {{r_off}}] {{row[:180]}}")
    header_info = f"Scanned {{total_scanned:,}} data rows | Showing {{len(hits)}} match(es):"
    return header_info + ("\\n" + "\\n".join(hits) if hits else " (none)")

def run(top_n=25):
    if not scanner_src:
        return "00_scan.py was not embedded in this archive."
    ns = {{}}
    exec(scanner_src, ns)
    return ns["run"](preloaded=loaded_parts, top_n=top_n)

run_scan = run

loaded_b = [f"INDEX_{{n:02d}}" for n in sorted(blocks_by_num)]
b_range = f"{{loaded_b[0]}}..{{loaded_b[-1]}}" if loaded_b else "none"
scan_msg = " | 00_scan.py READY: call run()" if scanner_src else ""
print(f"Unpacked {{len(loaded_parts)}}/{{total_expected}} standalone part(s): {{len(files_by_idx)}} entries across {{len(loaded_b)}} blocks ({{b_range}}){{scan_msg}}")"""

    scanner_note = (
        f"Embedded Scanner: ENABLED ({SCANNER_FILENAME} injected into every part; sandbox-safe fitz/numpy/json)\n"
        if embed_scan
        else "Embedded Scanner: DISABLED (run packpng.py with --embed-scan to include 00_scan.py)\n"
    )

    return (
        f"================================================================================\n"
        f"LLMPv2 SELF-CONTAINED IMAGE/PDF BUNDLE MANIFEST ({scale}x{scale} PIXEL SCALE)\n"
        f"================================================================================\n"
        f"Bundle ID     : llmp_{stamp}\n"
        f"Transport     : STANDALONE PARTS (each PDF/PNG decodes independently; no Part 1 dependency)\n"
        f"Pixel Geometry: {scale}x{scale} native RGB | Max Frame Capacity: {fmt_bytes(cap_bytes)} (@ {max_edge}x{max_edge})\n"
        f"Archive Stats : {len(entries):,} entries | {len(blocks)} logical blocks (<={block_kb}KB raw each) | {len(parts_info)} part(s)\n"
        f"Payload Size  : {total_raw:,} B raw (~{math.ceil(total_raw / 4):,} tokens) -> {total_packed:,} B packed ({ratio_pct:.1f}%, {'zlib' if compressed else 'raw'})\n"
        + scanner_note
        + "\n--- STANDALONE PART FILES (attach ONLY the Part needed for your target INDEX_XX block) ---\n"
        + "\n".join(part_lines)
        + "\n\n--- DIRECTORY TREE SUMMARY ---\n"
        + "\n".join(tree_lines)
        + f"\n\n--- LOGICAL BLOCK INDEX (<={block_kb}KB raw per block | ~125K tokens max per working set) ---\n"
        + "\n".join(block_lines)
        + "\n\n--- SANDBOX OPERATING PROTOCOL (META AI / GEMINI) ---\n"
        + "1. ZERO BANNED IMPORTS: The decoder below uses ONLY `fitz`, `numpy`, and `json` (decompression runs in C via\n"
        + "   MuPDF FlateDecode `xref_stream`, avoiding blocked `zlib`/`re`/`struct`/`glob`/`hashlib` imports).\n"
        + "2. KEEP STATE IN RAM: Once unpacked in turn 1, `files_by_idx`, `blocks_by_num`, and `loaded_parts` stay live in\n"
        + "   your Python session—run `run()` for entropy/dedupe/weight stats or query slices without re-decoding.\n"
        + "3. STDOUT PAGING: Always keep printed slices <= 18,000 chars (`get_block(n, page=1)` / `read_file(idx, page=1)`\n"
        + "   / `query_csv(query, col=...)`) so chat responses never truncate.\n"
        + "\n--- LLM SANDBOX DECODER (copy-paste into Python tool with attached Part PDF) ---\n"
        + f"{bt}python\n{decoder_snippet}\n{bt}\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Streaming low-RAM LLMPv2 packer (1x1 native RGB scale + auto CSV row-chunking + sandbox-safe decoder)."
    )
    ap.add_argument("-d", "--dir", default=".", help="Source directory to scan (default: current directory)")
    ap.add_argument("-o", "--out-dir", default=None, help="Output directory (default: same as --dir)")
    rec_group = ap.add_mutually_exclusive_group()
    rec_group.add_argument("-r", "--recursive", action="store_true", help="Include all subfolders recursively without prompting")
    rec_group.add_argument("--flat", action="store_true", help="Only scan top-level directory without prompting")
    ap.add_argument(
        "--embed-scan",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EMBED_SCANNER,
        help=f"Embed self-contained 00_scan.py analyzer into every part (default: {DEFAULT_EMBED_SCANNER})",
    )
    ap.add_argument("--ext", default="", help="Comma-separated extensions to include (e.g. .md,.json,.csv,.txt)")
    ap.add_argument("--all-text", action="store_true", help="Include any non-binary file regardless of extension")
    ap.add_argument("--no-compress", action="store_true", help="Disable zlib compression")
    ap.add_argument("--scale", type=int, choices=(1, 2, 4, 8), default=DEFAULT_SCALE, help=f"Pixel tile scale (default: {DEFAULT_SCALE} for 1x1 native RGB)")
    ap.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE, help=f"Max image edge in pixels (default: {DEFAULT_MAX_EDGE})")
    ap.add_argument("--max-part-mb", type=float, default=0.0, help="Optional compressed MB cap per part (default: auto max for --max-edge & --scale)")
    ap.add_argument("--block-kb", type=int, default=DEFAULT_BLOCK_KB, help=f"Max raw KB per logical index block & auto-chunk threshold (default: {DEFAULT_BLOCK_KB})")
    ap.add_argument("--max-raw", type=int, default=DEFAULT_MAX_RAW_MB, help=f"Max raw input MB guard (default: {DEFAULT_MAX_RAW_MB})")
    ap.add_argument("--clip", action="store_true", help="Copy manifest to Android clipboard via termux-clipboard-set")
    args = ap.parse_args()

    if args.max_edge <= HEADER_H + args.scale:
        print(f"[!] --max-edge must be greater than {HEADER_H + args.scale}")
        sys.exit(2)

    src_dir = os.path.abspath(args.dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else src_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.recursive:
        recursive = True
    elif args.flat:
        recursive = False
    else:
        subs = find_subfolders(src_dir)
        if subs:
            preview = ", ".join(subs[:4]) + (f" (+{len(subs) - 4} more)" if len(subs) > 4 else "")
            try:
                ans = input(f"Subfolders detected ({preview}). Do you want everything recursively added? [Y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return
            recursive = ans in ("y", "yes")
        else:
            recursive = False

    if args.all_text:
        exts = set()
    elif args.ext.strip():
        exts = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower() for e in args.ext.split(",")}
    else:
        exts = DEFAULT_EXTS

    prog = Progress()
    max_raw_bytes = max(1, args.max_raw) * 1_048_576
    max_block_bytes = max(16, args.block_kb) * 1024

    frame_cap = max_frame_capacity(args.max_edge, args.scale) - 12
    if args.max_part_mb > 0:
        max_payload_bytes = min(frame_cap, int(args.max_part_mb * 1_048_576))
    else:
        max_payload_bytes = frame_cap

    entries, total_raw = scan_files_streaming(
        src_dir, recursive, exts, os.path.basename(__file__), max_raw_bytes, max_block_bytes, args.embed_scan, prog
    )
    if not entries:
        print(f"[!] No matching text files found in: {src_dir}")
        return

    blocks = build_logical_blocks(entries, max_block_bytes)
    tree_lines = build_directory_tree(entries)
    compressed = not args.no_compress

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prelim_scanner = make_scanner_code([f"llmp_{stamp}_p99of99.pdf"]) if args.embed_scan else ""

    part_groups = plan_self_contained_parts(
        entries, blocks, compressed, max_payload_bytes, args.embed_scan, prelim_scanner, prog
    )

    parts_info = []
    num_parts = len(part_groups)
    all_pdf_names = [
        (f"llmp_{stamp}.pdf" if num_parts == 1 else f"llmp_{stamp}_p{k+1}of{num_parts}.pdf")
        for k in range(num_parts)
    ]
    final_scanner = make_scanner_code(all_pdf_names) if args.embed_scan else ""
    total_packed = 0

    for i, grp in enumerate(part_groups):
        part_idx = i + 1
        b_first, b_last = grp[0], grp[-1]
        block_span = (
            b_first["block_id"]
            if b_first["block_num"] == b_last["block_num"]
            else f"{b_first['block_id']}..{b_last['block_id']}"
        )
        file_span = f"{b_first['file_start']}..{b_last['file_end']}"

        prog.update(f"[3/4] Packing & verifying standalone part {part_idx}/{num_parts} ({block_span}, {args.scale}x)...", force=True)
        payload, part_raw_len, part_file_count = build_part_payload(
            entries, grp, part_idx, num_parts, compressed, args.embed_scan, final_scanner
        )
        total_packed += len(payload)

        frame = b"LLMP" + struct.pack(">BBBBI", 2, 1 if compressed else 0, part_idx, num_parts, len(payload)) + payload
        del payload

        file_preview = f"{block_span} | entries {file_span} ({part_file_count} entries, {fmt_bytes(part_raw_len)})"
        rgb_arr, sha, dim, verified = render_part_rgb(frame, part_idx, num_parts, {
            "max_edge": args.max_edge,
            "scale": args.scale,
            "compressed": compressed,
            "embed_scan": args.embed_scan,
            "part_raw_len": part_raw_len,
            "part_file_count": part_file_count,
            "block_span": block_span,
            "file_preview": file_preview,
        })
        frame_len = len(frame)
        del frame

        if not verified:
            raise RuntimeError(f"Self-test readback SHA-256 mismatch on part {part_idx}")

        stem = f"llmp_{stamp}" if num_parts == 1 else f"llmp_{stamp}_p{part_idx}of{num_parts}"
        png_path = os.path.join(out_dir, stem + ".png")
        pdf_path = os.path.join(out_dir, stem + ".pdf")

        Image.fromarray(rgb_arr, mode="RGB").save(png_path, format="PNG", optimize=False, compress_level=3, icc_profile=None)
        write_lossless_pdf(pdf_path, rgb_arr)
        del rgb_arr
        gc.collect()

        parts_info.append({
            "png": os.path.basename(png_path),
            "pdf": os.path.basename(pdf_path),
            "sha": sha,
            "dim": dim,
            "frame_len": frame_len,
            "block_span": block_span,
            "file_span": file_span,
            "part_files": part_file_count,
            "part_raw": part_raw_len,
        })

    prog.done(f"[3/4] Rendered & verified {num_parts} standalone PNG + PDF part(s) @ {args.scale}x (peak RAM: {peak_rss_mb():.1f} MB)")

    manifest_text = build_manifest(
        stamp, entries, blocks, tree_lines, total_raw, total_packed, compressed,
        parts_info, args.block_kb, args.scale, args.max_edge, args.embed_scan
    )
    manifest_path = os.path.join(out_dir, f"llmp_{stamp}_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        mf.write(manifest_text)

    if args.clip and shutil.which("termux-clipboard-set"):
        subprocess.run(["termux-clipboard-set"], input=manifest_text.encode("utf-8"), check=False)

    mode_lbl = "recursive" if recursive else "top-level"
    scan_lbl = ", +00_scan.py" if args.embed_scan else ""
    prog.done(
        f"[4/4] DONE ({mode_lbl}, {args.scale}x{scan_lbl}) | {len(entries)} entries -> {len(blocks)} blocks -> {num_parts} standalone part(s) | "
        f"Manifest: {fmt_bytes(len(manifest_text.encode('utf-8')))} | Peak RSS: {peak_rss_mb():.1f} MB"
    )
    for p in parts_info:
        print(f"  PNG : {os.path.join(out_dir, p['png'])} ({p['dim']['img_w']}x{p['dim']['img_h']}, {p['dim']['scale']}x, {p['block_span']})")
        print(f"  PDF : {os.path.join(out_dir, p['pdf'])} (WhatsApp-safe standalone)")
        print(f"  SHA : {p['sha']}")
    print(f"  TXT : {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        sys.exit(130)
    except MemoryError as me:
        print(f"\n[!] Memory safety guard triggered: {me}")
        sys.exit(1)
