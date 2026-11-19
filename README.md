# LLMPv2 + DLN1: Bidirectional Pixel Storage for Mobile AI

Turn standard images and single-page PDFs into a portable, serverless read/write cartridge drive for mobile LLM chat interfaces (Gemini, Meta AI on WhatsApp, ChatGPT).

![Sample LLMPv2 Cartridge](examples/sample_llmp_bundle.png)

## Why This Exists
Mobile AI chats impose two hard walls:
1. **Context Window Truncation:** Pasting large folders or multi-megabyte CSVs directly into chat cuts off data past the token limit.
2. **Aggressive Media Recompression:** Chat apps like WhatsApp resize outbound photos and re-encode them as lossy JPEGs (quality ~70%), destroying subtle pixel data.

This project solves both directions without external servers or APIs:
* **Inbound (`LLMPv2` — High-Density Read Drive):** Compresses text/CSV folders with `zlib`, paints raw bytes directly into `1×1` RGB pixels under a visual instruction banner, and wraps the raster in a minimal PDF 1.4 `/FlateDecode` container. Inside the AI's Python sandbox, MuPDF (`fitz.xref_stream`) inflates the payload in C—bypassing blocked `zlib`/`struct`/`re` imports—and mounts an embedded scanner (`00_scan.py`) in RAM.
* **Outbound (`DLN1` — Compression-Proof Write Drive):** Packs outputs from the AI sandbox into `8×8` pure-black (`0`) and pure-white (`255`) macro-tiles. Because a 255-step brightness gap survives WhatsApp JPEG recompression cleanly, a single-file HTML5 viewer (`viewer_dln1.html`) samples the tiles via your phone's GPU and downloads the rebuilt file in milliseconds.

## Verified Benchmarks

| Metric | Inbound (`LLMPv2` 1×1 RGB PDF) | Outbound (`DLN1` 8×8 1-Bit JPEG) |
| :--- | :--- | :--- |
| **Max Capacity / Frame** | ~45 MB raw text (`2048×2048`) | ~32.7 KB deflated payload (`4096×4096`) |
| **Transport Channel** | WhatsApp Document / Direct Upload | Standard WhatsApp Photo (survives JPEG) |
| **Sandbox Dependencies** | `fitz`, `numpy`, `json` only | Browser `<canvas>` + `DecompressionStream` |
| **Mobile Decode Time** | ~40 ms in C via `fitz.xref_stream` | < 150 ms across 6 frames on Android/iOS |

## 10-Second Browser Test (No Python Required)
1. Open the live browser decoder at `https://<your-username>.github.io/<repo-name>/viewer_dln1.html` (enable **GitHub Pages → Deploy from branch `main` / `(root)`** in repository settings), or download `viewer_dln1.html` and double-click it locally.
2. Download `examples/sample_dln1_whatsapp_sim.jpg` from this repository.
3. Select `sample_dln1_whatsapp_sim.jpg` inside the viewer—it decodes the JPEG-degraded 1-bit tiles, verifies the payload, and lets you download the reconstructed output immediately.

## Quickstart: Pack Your Own Files (Termux / Linux / macOS)

```bash
pip install -r requirements.txt
python3 packpng.py -d ./my_folder --recursive --clip
```

1. Attach the generated `llmp_<timestamp>.pdf` in your AI chat.
2. Paste the `LLM SANDBOX DECODER` snippet from `llmp_<timestamp>_manifest.txt`.
3. Inspect your mounted archive in RAM using `run()` (deduplication, Shannon entropy, information weight, and CSV profiling), `search_files("query")`, `query_csv("term", col="name")`, or `get_block(1)`.

## Reproduce the Proof Suite

```bash
python3 generate_test_suite.py
```

Generates a ~580 KB dummy corpus (`test_corpus/`), packs it into `examples/sample_llmp_bundle.pdf` and `examples/sample_llmp_bundle.png`, simulates WhatsApp JPEG recompression (`quality=70, subsampling=2`) on a `DLN1` export (`examples/sample_dln1_whatsapp_sim.jpg`), and verifies bit-exact `SHA-256` round-trip hashes in `examples/verification_report.json`.

## Security & Responsible Use
* **Treat Cartridges as Untrusted Input:** `LLMPv2` images can carry arbitrary text and embedded Python helpers (`00_scan.py`). Always audit embedded scripts before executing them in shared environments.
* **Authorized Testing Only:** Because image-wrapped payloads bypass plain-text content filters, use this architecture only on data and systems you own or have explicit permission to test.
