#!/usr/bin/env python3
"""
generate_test_suite.py — Deterministic Corpus Generator & Round-Trip Proof Suite
Creates generic dummy text/CSV/JSON files in ./test_corpus/, packs them into LLMPv2
(1x1 RGB + PDF 1.4) and DLN1 (8x8 1-bit WhatsApp-degraded JPEG) artifacts in
./examples/, and verifies bit-exact SHA-256 recovery in both directions.
"""

import csv
import hashlib
import json
import math
import os
import struct
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_dummy_corpus(corpus_dir: str) -> dict:
    os.makedirs(corpus_dir, exist_ok=True)

    # 1. Generic multi-chunk dummy CSV (>560 KB so it exercises >500 KB block splitting)
    csv_path = os.path.join(corpus_dir, "dummy_telemetry.csv")
    nodes = [
        ("NODE_ALPHA", "us-west", "NOMINAL", "Standard packet relay heartbeat verified."),
        ("NODE_BRAVO", "eu-central", "WARN", "Buffer latency exceeded 120ms threshold."),
        ("NODE_CHARLIE", "ap-southeast", "NOMINAL", "Checksum validation completed with zero errors."),
        ("NODE_DELTA", "us-east", "INFO", "Scheduled snapshot rotation archived to disk."),
        ("NODE_ECHO", "sa-east", "ERROR", "Upstream socket reset during frame sync."),
    ]
    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["record_id", "node", "region", "level", "latency_ms", "message"])
        while os.path.getsize(csv_path) < 560 * 1024:
            node, region, level, msg = nodes[rows_written % len(nodes)]
            latency = round(12.5 + ((rows_written * 37) % 240) * 0.75, 2)
            writer.writerow([
                f"REC-{rows_written + 1:05d}",
                node,
                region,
                level,
                latency,
                f"{msg} [seq={rows_written + 1} | crc={(rows_written * 7919) % 100000:05d}]",
            ])
            rows_written += 1

    # 2. Generic Markdown notes
    md_path = os.path.join(corpus_dir, "notes_architecture.md")
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(
            "# LLMPv2 + DLN1 Architecture Notes\n\n"
            "## Inbound Channel (LLMPv2)\n"
            "- Compresses text files with zlib and maps bytes directly to 1x1 RGB pixels.\n"
            "- Wraps the raster inside a PDF 1.4 /FlateDecode Image XObject to bypass WhatsApp transcoding.\n\n"
            "## Outbound Channel (DLN1)\n"
            "- Encodes deflated JSON results into 8x8 black (0) and white (255) macro-tiles.\n"
            "- Survives WhatsApp JPEG recompression (quality ~70) cleanly via 128 luma thresholding.\n"
        )

    # 3. Generic JSON config
    json_path = os.path.join(corpus_dir, "config_rules.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "protocol": "LLMPv2-DLN1",
            "inbound_scale": "1x1_RGB_PDF",
            "outbound_scale": "8x8_1BIT_JPEG",
            "whatsapp_threshold": 128,
            "target_records": rows_written,
        }, jf, indent=2)

    return {
        "csv_path": csv_path,
        "csv_rows": rows_written,
        "csv_bytes": os.path.getsize(csv_path),
        "md_bytes": os.path.getsize(md_path),
        "json_bytes": os.path.getsize(json_path),
    }


def write_lossless_pdf(pdf_path: str, rgb_arr: np.ndarray):
    h, w, _ = rgb_arr.shape
    comp_rgb = zlib.compress(rgb_arr.tobytes(), level=3)

    o1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    o2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    o3 = (
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] "
        f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    ).encode("ascii")
    o4h = (
        f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(comp_rgb)} >>\nstream\n"
    ).encode("ascii")
    o4f = b"\nendstream\nendobj\n"
    cs = f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q\n".encode("ascii")
    o5 = f"5 0 obj\n<< /Length {len(cs)} >>\nstream\n".encode("ascii") + cs + b"endstream\nendobj\n"

    phdr = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offs, pos = [], len(phdr)
    for sz in (len(o1), len(o2), len(o3), len(o4h) + len(comp_rgb) + len(o4f), len(o5)):
        offs.append(pos)
        pos += sz

    with open(pdf_path, "wb") as pf:
        pf.write(phdr + o1 + o2 + o3 + o4h + comp_rgb + o4f + o5)
        pf.write(b"xref\n0 6\n0000000000 65535 f \n")
        for o in offs:
            pf.write(f"{o:010d} 00000 n \n".encode("ascii"))
        pf.write(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n".encode("ascii"))


def verify_llmp_inbound(corpus_dir: str, examples_dir: str) -> dict:
    raw_files = []
    for fn in sorted(os.listdir(corpus_dir)):
        fp = os.path.join(corpus_dir, fn)
        if os.path.isfile(fp):
            raw_files.append((fn, open(fp, "rb").read()))

    toc_entries = []
    blob_parts = []
    off = 0
    for idx, (fn, data) in enumerate(raw_files):
        toc_entries.append({
            "global_idx": idx,
            "name": fn,
            "offset": off,
            "len": len(data),
            "sha256": sha256_bytes(data),
        })
        blob_parts.append(data)
        off += len(data)

    toc_bytes = json.dumps(
        {"schema": "LLMPv2", "part": 1, "total_parts": 1, "files": toc_entries},
        separators=(",", ":"),
    ).encode("utf-8")
    stream = struct.pack(">I", len(toc_bytes)) + toc_bytes + b"".join(blob_parts)
    comp = zlib.compress(stream, level=6)
    frame = b"LLMP" + struct.pack(">BBBBI", 2, 1, 1, 1, len(comp)) + comp
    frame_sha = sha256_bytes(frame)

    header_h = 128
    img_w = 512
    logical_px = math.ceil(len(frame) / 3)
    grid_h = max(32, math.ceil(logical_px / img_w))
    img_h = header_h + grid_h

    padded = np.zeros(img_w * grid_h * 3, dtype=np.uint8)
    padded[:len(frame)] = np.frombuffer(frame, dtype=np.uint8)
    grid = padded.reshape((grid_h, img_w, 3))

    hdr = Image.new("RGB", (img_w, header_h), (255, 212, 0))
    draw = ImageDraw.Draw(hdr)
    cal_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)]
    for i, col in enumerate(cal_colors):
        x0 = (img_w - 114) + i * 21
        draw.rectangle([x0, 8, x0 + 17, 25], fill=col, outline=(0, 0, 0) if col == (255, 255, 255) else None)

    draw.text((10, 10), f"LLMPv2 (1x1 NATIVE) | PART=1/1 | COMPRESS=zlib | GRID={img_w}x{grid_h}", fill=(0, 0, 0))
    draw.text((10, 32), f"RAW={off} | FRAME_LEN={len(frame)} | FILES={len(raw_files)} | IMG={img_w}x{img_h}", fill=(0, 0, 0))
    draw.text((10, 54), f"SHA256={frame_sha}", fill=(0, 0, 0))
    draw.text((10, 76), "DECODE: Pixmap RGB[y>=128], bytes[:FRAME_LEN], skip 12B hdr, inflate, parse TOC", fill=(0, 0, 0))
    draw.text((10, 98), f"CONTENTS: {', '.join(fn for fn, _ in raw_files)}", fill=(0, 0, 0))

    full_rgb = np.vstack([np.array(hdr, dtype=np.uint8), grid])
    png_path = os.path.join(examples_dir, "sample_llmp_bundle.png")
    pdf_path = os.path.join(examples_dir, "sample_llmp_bundle.pdf")
    Image.fromarray(full_rgb, mode="RGB").save(png_path, format="PNG")
    write_lossless_pdf(pdf_path, full_rgb)

    if HAS_FITZ:
        doc = fitz.open(pdf_path)
        imgs = doc.get_page_images(0)
        pix = fitz.Pixmap(doc, imgs[0][0])
        read_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))[:, :, :3]
        doc.close()
        decode_path = "fitz.Pixmap (PDF XObject)"
    else:
        read_arr = np.array(Image.open(png_path).convert("RGB"))
        decode_path = "PIL.Image (PNG fallback)"

    rec_frame = read_arr[header_h:, :, :].tobytes()[:len(frame)]
    assert sha256_bytes(rec_frame) == frame_sha, "LLMPv2 frame SHA-256 mismatch!"
    clen = struct.unpack(">I", rec_frame[8:12])[0]
    rec_stream = zlib.decompress(rec_frame[12 : 12 + clen])
    tlen = struct.unpack(">I", rec_stream[:4])[0]
    rec_toc = json.loads(rec_stream[4 : 4 + tlen].decode("utf-8"))
    rec_blob = rec_stream[4 + tlen :]

    for entry in rec_toc["files"]:
        piece = rec_blob[entry["offset"] : entry["offset"] + entry["len"]]
        assert sha256_bytes(piece) == entry["sha256"], f"File hash mismatch: {entry['name']}"

    return {
        "raw_total_bytes": off,
        "compressed_frame_bytes": len(frame),
        "compression_ratio_pct": round(100.0 * len(frame) / off, 2),
        "image_dimensions": f"{img_w}x{img_h}",
        "pdf_bytes": os.path.getsize(pdf_path),
        "png_bytes": os.path.getsize(png_path),
        "frame_sha256": frame_sha,
        "verified_via": decode_path,
        "all_file_hashes_match": True,
    }


def verify_dln1_outbound(corpus_info: dict, examples_dir: str) -> dict:
    sample_report = {
        "suite": "LLMPv2 -> DLN1 Outbound Export Verification",
        "source_rows": corpus_info["csv_rows"],
        "source_bytes": corpus_info["csv_bytes"] + corpus_info["md_bytes"] + corpus_info["json_bytes"],
        "sample_records": [
            {"record_id": "REC-00001", "node": "NODE_ALPHA", "region": "us-west", "level": "NOMINAL", "latency_ms": 12.5},
            {"record_id": "REC-00002", "node": "NODE_BRAVO", "region": "eu-central", "level": "WARN", "latency_ms": 40.25},
            {"record_id": "REC-00003", "node": "NODE_CHARLIE", "region": "ap-southeast", "level": "NOMINAL", "latency_ms": 68.0},
            {"record_id": "REC-00004", "node": "NODE_DELTA", "region": "us-east", "level": "INFO", "latency_ms": 95.75},
            {"record_id": "REC-00005", "node": "NODE_ECHO", "region": "sa-east", "level": "ERROR", "latency_ms": 123.5},
        ],
    }

    json_str = json.dumps(sample_report, indent=2)
    json_bytes = json_str.encode("utf-8")
    json_sha = sha256_bytes(json_bytes)

    cobj = zlib.compressobj(level=6, wbits=-15)
    raw_def = cobj.compress(json_bytes) + cobj.flush()
    dln1_payload = b"DLN1" + struct.pack(">I", len(raw_def)) + raw_def

    bits = np.unpackbits(np.frombuffer(dln1_payload, dtype=np.uint8))
    side_tiles = 64
    while (side_tiles - 1) * side_tiles < len(bits):
        side_tiles += 16

    tile_grid = np.zeros((side_tiles, side_tiles), dtype=np.uint8)
    tile_grid[0, 1::2] = 255
    flat_data = np.zeros((side_tiles - 1) * side_tiles, dtype=np.uint8)
    flat_data[:len(bits)] = bits * 255
    tile_grid[1:, :] = flat_data.reshape((side_tiles - 1, side_tiles))

    tile_px = 8
    grid_px = np.repeat(np.repeat(tile_grid, tile_px, axis=0), tile_px, axis=1)
    rgb = np.stack([grid_px, grid_px, grid_px], axis=-1)

    jpg_path = os.path.join(examples_dir, "sample_dln1_whatsapp_sim.jpg")
    Image.fromarray(rgb, mode="RGB").save(jpg_path, format="JPEG", quality=70, subsampling=2)

    degraded = np.array(Image.open(jpg_path).convert("RGB"), dtype=np.float32)
    luma = 0.299 * degraded[:, :, 0] + 0.587 * degraded[:, :, 1] + 0.114 * degraded[:, :, 2]
    centers = luma.reshape(side_tiles, tile_px, side_tiles, tile_px)[:, 2:6, :, 2:6].mean(axis=(1, 3))
    rec_bits = (centers[1:, :].flatten() > 128).astype(np.uint8)
    rec_bytes = np.packbits(rec_bits).tobytes()

    assert rec_bytes[:4] == b"DLN1", "DLN1 magic corrupted after JPEG recompression!"
    r_clen = struct.unpack(">I", rec_bytes[4:8])[0]
    rec_json_bytes = zlib.decompress(rec_bytes[8 : 8 + r_clen], wbits=-15)
    assert sha256_bytes(rec_json_bytes) == json_sha, "DLN1 JSON SHA-256 mismatch!"

    return {
        "original_json_bytes": len(json_bytes),
        "deflated_bytes": len(raw_def),
        "tile_grid": f"{side_tiles}x{side_tiles} (8x8 px tiles = {side_tiles * tile_px}x{side_tiles * tile_px} px)",
        "simulated_whatsapp_jpeg_bytes": os.path.getsize(jpg_path),
        "jpeg_compression_settings": "quality=70, subsampling=2 (4:2:0)",
        "json_sha256": json_sha,
        "bit_exact_recovery_after_jpeg": True,
    }


def main():
    t0 = time.monotonic()
    root = os.path.abspath(".")
    corpus_dir = os.path.join(root, "test_corpus")
    examples_dir = os.path.join(root, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    corpus_info = build_dummy_corpus(corpus_dir)
    inbound_proof = verify_llmp_inbound(corpus_dir, examples_dir)
    outbound_proof = verify_dln1_outbound(corpus_info, examples_dir)

    report = {
        "suite": "LLMPv2 + DLN1 Round-Trip Verification",
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "corpus": corpus_info,
        "inbound_llmpv2": inbound_proof,
        "outbound_dln1": outbound_proof,
    }

    report_path = os.path.join(examples_dir, "verification_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

