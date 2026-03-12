from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import io

def load_image(pil_image):
    """Convert input image to OpenCV BGR numpy array."""
    if not isinstance(pil_image, Image.Image):
        with Image.open(pil_image) as img:
            rgb = img.convert("RGB")
    else:
        rgb = pil_image.convert("RGB")

    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(pil_image, max_dim=1600):
    """Resize, denoise, and enhance contrast for OCR."""
    try:
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.open(pil_image)

        w, h = pil_image.size
        scale = min(1.0, float(max_dim) / max(w, h))

        if scale < 1.0:
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        gray = pil_image.convert("L")
        enhanced = ImageOps.autocontrast(gray)
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3)
        )

        return enhanced

    except Exception:
        return pil_image.convert("L")


def to_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()

    if fmt.upper() == "JPEG":
        pil_image = pil_image.convert("RGB")

    pil_image.save(buf, format=fmt)
    buf.seek(0)

    return buf.getvalue()