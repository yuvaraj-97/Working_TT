# sdc.py
import os, re, glob, argparse, json
from statistics import median
from pathlib import Path
from typing import Optional, Sequence

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract, cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========= DEFAULT CONFIG (you can override via CLI) =========
PDF_DIR_DEFAULT = r"C:\Users\U215438\OneDrive - Trane Technologies\Documents\Python\Yuvaraj_A\EVA_API\PDF"
OUTPUT_DIR_DEFAULT = r"C:\Users\U215438\OneDrive - Trane Technologies\Documents\Python\Yuvaraj_A\EVA_API\output"
TESSERACT_EXE_DEFAULT = r"C:\Users\U215438\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_BIN_DEFAULT = r"C:\Users\U215438\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
LANG_DEFAULT = "eng"
# =============================================================

# Make Pillow safe for huge engineering scans (trusted local files)
Image.MAX_IMAGE_PIXELS = 1_000_000_000

# ------------------- Helpers -------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _validate_pil_image(pil_img: Image.Image, context: str = ""):
    """Raise ValueError if the PIL image is empty (width/height == 0)."""
    if pil_img is None:
        raise ValueError(f"Empty image supplied{': ' + context if context else ''}")
    w, h = getattr(pil_img, "width", 0), getattr(pil_img, "height", 0)
    if not w or not h:
        raise ValueError(f"Zero-sized image encountered{': ' + context if context else ''}")


def pil2cv(img: Image.Image, context: str = ""):
    _validate_pil_image(img, context)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.size == 0:
        raise ValueError(f"Image data empty after conversion{': ' + context if context else ''}")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2pil(img):
    if img is None or img.size == 0:
        raise ValueError("Empty OpenCV image supplied")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def preprocess_for_ocr(pil_img: Image.Image, context: str = "") -> Image.Image:
    """General cleaning that works well on scans/drawings/tables."""
    try:
        cv = pil2cv(pil_img, context=context)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 15)
        return cv2pil(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    except (ValueError, cv2.error) as exc:
        msg_ctx = f" ({context})" if context else ""
        print(f"[WARN] Skipping preprocessing{msg_ctx}: {exc}")
        return pil_img.convert("RGB")

def autorotate(pil_img: Image.Image) -> Image.Image:
    """Use Tesseract OSD to correct rotation when OCR’ing."""
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        angle = int(m.group(1)) if m else 0
        return pil_img.rotate(-angle, expand=True, fillcolor="white") if angle else pil_img
    except Exception:
        return pil_img

def page_profile(page: fitz.Page):
    """
    Inspect a page with PyMuPDF:
      has_text: bool (nontrivial text layer?)
      img_coverage: fraction of page covered by image blocks (0..1)
      median_font_pt: median font size (pt) if any text spans, else None
    """
    w, h = page.rect.width, page.rect.height
    area = max(w * h, 1.0)
    rd = page.get_text("rawdict")

    text_chars = 0
    font_sizes = []
    img_area = 0.0

    for block in rd["blocks"]:
        btype = block.get("type", 0)
        bbox = block.get("bbox", (0,0,0,0))
        bw = max(0.0, bbox[2] - bbox[0])
        bh = max(0.0, bbox[3] - bbox[1])

        if btype == 0:  # text
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_chars += len(span.get("text", ""))
                    fs = span.get("size")
                    if fs:
                        font_sizes.append(fs)
        elif btype == 1:  # image
            img_area += (bw * bh)

    has_text = text_chars >= 20
    med_font = median(font_sizes) if font_sizes else None
    coverage = img_area / area
    return has_text, coverage, med_font

def choose_dpi(has_text: bool, img_cov: float, median_pt: float | None) -> int:
    """Adaptive DPI policy per page."""
    if not has_text or img_cov >= 0.90:
        return 400  # image-only / dominant scan
    if median_pt is not None and median_pt < 8.5:
        return 350  # tiny fonts
    if median_pt is not None and median_pt > 13:
        return 250  # big fonts
    return 300  # normal text

def ocr_image(pil_img: Image.Image, lang: str, table_mode=False, context: str = "") -> str:
    pil_img = autorotate(pil_img)
    pil_img = preprocess_for_ocr(pil_img, context=context or "ocr_image")
    if table_mode:
        cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./-()+°%,±Øø'
    else:
        cfg = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(pil_img, lang=lang, config=cfg)

# ------ Basic grid/table detector (best effort) ------
def extract_table_to_csv(pil_img: Image.Image, lang: str, out_csv: Path, context: str = "") -> tuple[bool, list[list[str]]]:
    """Detect grid lines, OCR cells, and write CSV.

    Returns
    -------
    tuple[bool, list[list[str]]]
        A flag indicating whether a table was detected along with the table
        contents (each inner list represents one row).
    """
    ensure_dir(out_csv.parent)
    ctx = context or f"table:{out_csv.name}"
    cv = pil2cv(preprocess_for_ocr(pil_img, context=ctx), context=ctx)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    H, W = inv.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 60), 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, H // 40)))

    horiz = cv2.dilate(cv2.erode(inv, horiz_kernel, 1), horiz_kernel, 1)
    vert  = cv2.dilate(cv2.erode(inv, vert_kernel, 1),  vert_kernel, 1)
    grid  = cv2.add(horiz, vert)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 200:       # skip tiny
            continue
        if w > W*0.98 and h > H*0.15:  # skip large outer borders
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return False, []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    # group into rows
    rows, current, row_y = [], [boxes[0]], boxes[0][1]
    row_tol = max(10, H // 100)
    for b in boxes[1:]:
        if abs(b[1] - row_y) <= row_tol:
            current.append(b)
        else:
            rows.append(sorted(current, key=lambda x: x[0]))
            current, row_y = [b], b[1]
    rows.append(sorted(current, key=lambda x: x[0]))

    col_count = max(len(r) for r in rows)
    cfg_whitelist = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./-()+°%±Øø,'
    table = []
    for r in rows:
        if len(r) < col_count:
            r = r + [r[-1]]*(col_count - len(r))
        row_text = []
        for (x,y,w,h) in r[:col_count]:
            pad = 2
            x0,y0,x1,y1 = max(0,x+pad), max(0,y+pad), min(W,x+w-pad), min(H,y+h-pad)
            cell = cv2pil(cv[y0:y1, x0:x1])
            txt = pytesseract.image_to_string(cell, lang=lang, config=cfg_whitelist)
            row_text.append(txt.strip().replace("\n", " "))
        table.append(row_text)

    pd.DataFrame(table).to_csv(out_csv, index=False, header=False)
    return True, table

# ------------------- Per-page runner -------------------
def render_pdf_page(pdf_path: Path, page_index: int, dpi: int, poppler_bin: str | None) -> Image.Image | None:
    """Render a single PDF page into a PIL image, handling empty results safely."""
    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_index + 1,
            last_page=page_index + 1,
            dpi=dpi,
            poppler_path=poppler_bin,
        )
    except Exception as exc:
        print(f"[ERROR] {pdf_path.name} page {page_index+1}: convert_from_path failed ({exc})")
        return None

    if not images:
        print(f"[WARN] {pdf_path.name} page {page_index+1}: no rasterized images returned")
        return None

    img = images[0]
    try:
        _validate_pil_image(img, context=f"{pdf_path.name}#p{page_index+1}")
    except ValueError as exc:
        print(f"[WARN] {pdf_path.name} page {page_index+1}: {exc}")
        return None
    return img


def process_page(doc, pdf_path: Path, page_index: int, out_dir: Path, lang: str,
                 poppler_bin: str | None, tiles=(3,3), tile_overlap=0.04):
    page = doc[page_index]
    has_text, img_cov, med_pt = page_profile(page)
    dpi = choose_dpi(has_text, img_cov, med_pt)

    page_txt = out_dir / f"page_{page_index+1:03d}.txt"
    page_csv = out_dir / f"page_{page_index+1:03d}_table.csv"

    page_info: dict[str, object] = {
        "page_number": page_index + 1,
        "dpi": dpi,
        "image_coverage": img_cov,
        "mode": None,
        "raw_text": None,
        "ocr_text": None,
        "table": None,
        "notes": [],
    }

    if has_text and img_cov < 0.90:
        # Prefer native text; OCR image regions if worthwhile
        txt_native = page.get_text("raw").strip()
        if img_cov > 0.15:
            pil = render_pdf_page(pdf_path, page_index, dpi, poppler_bin)
            if pil is None:
                print(f"[WARN] Skipping OCR image regions for {pdf_path.name} page {page_index+1}: could not render image")
                page_txt.write_text(txt_native, encoding="utf-8")
                page_info["mode"] = "native"
                page_info["raw_text"] = txt_native
                page_info["notes"].append("image-region-render-failed")
                return dpi, "native", img_cov, page_info
            W,H = pil.size
            img_region = pil.crop((0, int(H*0.4), W, H))
            made_csv, table_rows = extract_table_to_csv(img_region, lang, page_csv,
                                                        context=f"{pdf_path.name}#p{page_index+1}-img")
            if made_csv:
                page_info["table"] = {
                    "csv": page_csv.name,
                    "rows": table_rows,
                }
            else:
                ocr_txt = ocr_image(img_region, lang, table_mode=True,
                                    context=f"{pdf_path.name}#p{page_index+1}-img").strip()
                page_info["ocr_text"] = ocr_txt
        page_txt.write_text(txt_native, encoding="utf-8")
        page_info["mode"] = "native+ocr" if img_cov > 0.15 else "native"
        page_info["raw_text"] = txt_native
        return dpi, page_info["mode"], img_cov, page_info

    # Image-only / dominant → OCR with tiling + try table CSV
    pil = render_pdf_page(pdf_path, page_index, dpi, poppler_bin)
    if pil is None:
        print(f"[ERROR] {pdf_path.name} page {page_index+1}: failed to rasterize for OCR")
        page_info["mode"] = "render-error"
        page_info["notes"].append("render-failed")
        return dpi, "render-error", img_cov, page_info
    pil = autorotate(pil)
    pil_clean = preprocess_for_ocr(pil, context=f"{pdf_path.name}#p{page_index+1}")

    made_csv, table_rows = extract_table_to_csv(pil_clean, lang, page_csv,
                                                context=f"{pdf_path.name}#p{page_index+1}")
    if made_csv:
        page_info["table"] = {
            "csv": page_csv.name,
            "rows": table_rows,
        }

    W, H = pil_clean.size
    nx, ny = tiles
    tw, th = int(W/nx), int(H/ny)
    dx, dy = int(tw*tile_overlap), int(th*tile_overlap)

    tile_texts = []
    with tqdm(total=nx*ny, desc=f"  tiles@p{page_index+1}", leave=False) as pbar_tiles:
        for iy in range(ny):
            for ix in range(nx):
                x0 = max(0, ix*tw - dx); y0 = max(0, iy*th - dy)
                x1 = min(W, (ix+1)*tw + dx); y1 = min(H, (iy+1)*th + dy)
                tile = pil_clean.crop((x0,y0,x1,y1))
                cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./-()+°%,±Øø'
                txt = pytesseract.image_to_string(tile, lang=lang, config=cfg)
                tile_texts.append(f"[tile {ix},{iy}]\n{txt.strip()}\n")
                pbar_tiles.update(1)

    ocr_text = "\n".join(tile_texts)
    page_txt.write_text(ocr_text, encoding="utf-8")
    page_info["mode"] = "ocr-tiled"
    page_info["ocr_text"] = ocr_text.strip()
    page_info["raw_text"] = None
    return dpi, "ocr-tiled", img_cov, page_info

# ------------------- PDF runner -------------------
def process_pdf(pdf_path: Path, out_root: Path, lang: str, poppler_bin: str | None):
    out_dir = out_root / pdf_path.stem
    ensure_dir(out_dir)
    page_summaries = []
    page_infos = []

    try:
        with fitz.open(str(pdf_path)) as doc:
            if len(doc) == 0:
                print(f"[WARN] Empty PDF: {pdf_path.name}")
                return
            pages_iter = tqdm(range(len(doc)), desc=f"pages: {pdf_path.name}", leave=False)
            for p in pages_iter:
                try:
                    dpi, mode, imgcov, page_info = process_page(doc, pdf_path, p, out_dir, lang, poppler_bin)
                    page_summaries.append((p+1, dpi, mode, imgcov))
                    page_infos.append(page_info)
                except Exception as e:
                    (out_dir / "errors.log").open("a", encoding="utf-8").write(f"page {p+1}: {e}\n")
                    print(f"[ERROR] {pdf_path.name} page {p+1}: {e}")
                    page_infos.append({
                        "page_number": p + 1,
                        "dpi": None,
                        "image_coverage": None,
                        "mode": "error",
                        "raw_text": None,
                        "ocr_text": None,
                        "table": None,
                        "notes": [f"exception: {e}"],
                    })
    except Exception as e:
        print(f"[FATAL] Could not open {pdf_path}: {e}")
        return

    # Combine text & save manifest
    combined = []
    for p, _, _, _ in page_summaries:
        pt = out_dir / f"page_{p:03d}.txt"
        if pt.exists():
            combined.append(f"\n===== PAGE {p} =====\n")
            combined.append(pt.read_text(encoding="utf-8", errors="ignore"))
    if combined:
        (out_dir / "combined.txt").write_text("".join(combined), encoding="utf-8")

    pd.DataFrame(page_summaries, columns=["page", "dpi", "mode", "image_coverage"])\
      .to_csv(out_dir / "manifest.csv", index=False)

    storage_doc = {
        "document": pdf_path.name,
        "pages": page_infos,
    }
    (out_dir / "storage.json").write_text(
        json.dumps(storage_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ------------------- CLI / Main -------------------
def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse CLI args but gracefully ignore stray ones from notebook launchers."""

    ap = argparse.ArgumentParser(description="Smart PDF text extractor (adaptive OCR).")
    ap.add_argument("--pdf-dir", default=PDF_DIR_DEFAULT, help="Folder containing PDFs.")
    ap.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output folder.")
    ap.add_argument("--tesseract", default=TESSERACT_EXE_DEFAULT, help="Path to tesseract.exe on Windows.")
    ap.add_argument("--poppler", default=POPPLER_BIN_DEFAULT, help="Path to Poppler bin (pdfinfo.exe, pdftoppm.exe).")
    ap.add_argument("--lang", default=LANG_DEFAULT, help="Tesseract languages, e.g. 'eng' or 'eng+deu'.")

    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"[WARN] Ignoring unrecognized arguments: {unknown}")
    return args

def main():
    args = parse_args()

    pytesseract.pytesseract.tesseract_cmd = args.tesseract
    poppler_bin = args.poppler if args.poppler and len(args.poppler.strip()) else None

    pdf_dir = Path(args.pdf_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)

    print(f"[INFO] cwd: {Path.cwd()}")
    print(f"[INFO] PDF_DIR: {pdf_dir}")
    print(f"[INFO] OUTPUT_DIR: {out_dir}")
    print(f"[INFO] POPPLER_BIN: {poppler_bin}")
    print(f"[INFO] Tesseract: {pytesseract.pytesseract.tesseract_cmd}")

    # List contents (helps with OneDrive placeholders)
    try:
        names = [p.name for p in pdf_dir.iterdir()]
        print(f"[INFO] PDF_DIR exists: {pdf_dir.exists()}  items: {len(names)}  sample: {names[:10]}")
    except Exception as e:
        print(f"[WARN] Could not list PDF_DIR: {e}")

    pdf_files = sorted(glob.glob(str(pdf_dir / "*.pdf"))) + sorted(glob.glob(str(pdf_dir / "*.PDF")))
    print(f"[INFO] PDFs found: {len(pdf_files)} -> {[Path(p).name for p in pdf_files]}")

    if not pdf_files:
        print("[HINT] Put PDFs in the folder above OR pass --pdf-dir FULL_PATH.")
        print("[HINT] In OneDrive, right-click PDFs → 'Always keep on this device'.")
        return

    for pdf in tqdm(pdf_files, desc="PDFs"):
        process_pdf(Path(pdf), out_dir, args.lang, poppler_bin)

    print(f"\nDone. See results in: {out_dir}")

if __name__ == "__main__":
    main()