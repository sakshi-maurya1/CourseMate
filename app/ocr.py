import cv2
import fitz  # PyMuPDF
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from app import config

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def render_page(page, dpi=None):
    pix = page.get_pixmap(dpi=dpi or config.OCR_DPI, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def deskew(gray):
    inv = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(inv)
    if pts is None:
        return gray
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    if not 0.3 < abs(angle) < 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def ocr_image(img, psm=None):
    """Returns (text, mean confidence 0-100). psm kept only for compatibility."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result, _ = engine()(img)
    if not result:
        return "", 0.0
    texts = [r[1] for r in result]
    confs = [float(r[2]) for r in result]
    return " ".join(texts), 100 * sum(confs) / len(confs)


def ocr_page(page):
    return ocr_image(deskew(render_page(page)))


def figure_text(doc, page, min_side=150):
    out = []
    for info in page.get_images(full=True):
        try:
            raw = doc.extract_image(info[0])["image"]
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        if img is None or min(img.shape[:2]) < min_side:
            continue
        text, conf = ocr_image(img)
        if len(text) >= 8 and conf >= 50:
            out.append(text)
    return " | ".join(out) 