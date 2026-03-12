"""OCR and LLM conversion helpers."""

import os
import logging
import re
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------- OCR IMPORTS ---------------- #

try:
    import easyocr
    _have_easyocr = True
except Exception:
    _have_easyocr = False

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _have_tesseract = True
except Exception:
    _have_tesseract = False

try:
    from openai import OpenAI
    _have_openai = True
except Exception:
    _have_openai = False

# ---------------- OCR ENGINE ---------------- #

class OCREngine:

    def __init__(self, lang_list=["en"]):

        self.lang_list = lang_list
        self.reader = None

        if _have_easyocr:
            try:
                self.reader = easyocr.Reader(lang_list, gpu=False)
            except Exception:
                logger.exception("EasyOCR initialization failed")

    # ------------------------------------------

    def extract_text(self, pil_image):

        """Extract raw text from the image using available OCR engines."""

        try:

            if self.reader:
                result = self.reader.readtext(np.asarray(pil_image))
                texts = [r[1] for r in result]
                return "\n".join(texts)

            if _have_tesseract:
                return pytesseract.image_to_string(pil_image)

            return ""

        except Exception as e:
            logger.exception("OCR extraction failed: %s", e)
            return ""

    # ------------------------------------------

    def extract_math(self, pil_image):

        """Return cleaned math expression from OCR."""

        raw = self.extract_text(pil_image)

        try:
            math = _extract_math_from_text(raw)

            if math:
                return math

        except Exception:
            logger.exception("Math extraction failed")

        return raw


# ---------------- TEXT CLEANING ---------------- #

def _clean_ocr_text(ocr_text: str) -> str:

    if not ocr_text:
        return ""

    text = ocr_text.strip()

    text = text.replace("×", "*")
    text = text.replace("—", "-")

    text = re.sub(r"[^0-9a-zA-Z\s=+\-*/().^]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------- MATH EXTRACTION ---------------- #

def _extract_math_from_text(ocr_text: str) -> str:

    if not ocr_text:
        return ""

    allowed = set(
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=+-*/^()[]{} ._%"
    )

    math_lines = []

    for line in ocr_text.splitlines():

        if not line.strip():
            continue

        filtered = "".join(ch for ch in line if ch in allowed)

        filtered = re.sub(
            r"\b(Algebra|Problems|Problem|Exercises)\b",
            "",
            filtered,
            flags=re.IGNORECASE,
        )

        filtered = re.sub(r"\s+", " ", filtered).strip()

        if filtered:
            math_lines.append(filtered)

    if not math_lines:
        return _clean_ocr_text(ocr_text)

    def score(s):

        math_symbols = sum(1 for ch in s if ch in "=+-*/^()[]{}")

        return len(s) + math_symbols * 3

    best = max(math_lines, key=score)

    best = best.replace("\u00bd", "1/2")

    try:

        nospace = re.sub(r"\s+", "", best)

        if re.search(r"[+\-*/^=]", nospace) and re.search(r"\d", nospace):
            return nospace

        candidates = re.findall(r"[0-9A-Za-z^/*+\-()=]+", best)

        def is_math_like(c):

            return bool(re.search(r"[+\-*/^=]", c) and re.search(r"\d", c))

        math_cands = [c for c in candidates if is_math_like(c)]

        if math_cands:

            return max(
                math_cands,
                key=lambda x: (len(x), sum(1 for ch in x if ch in "+-*/^=")),
            )

    except Exception:
        pass

    return best


# ---------------- GEMINI API ---------------- #

def _call_gemini_api(prompt: str, api_key: str):

    try:

        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-bison-001:generateText?key={api_key}"

        headers = {"Content-Type": "application/json"}

        body = {
            "prompt": {"text": prompt},
            "temperature": 0.0,
            "maxOutputTokens": 512,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=20)

        resp.raise_for_status()

        data = resp.json()

        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0].get("output", "")

        return ""

    except Exception as e:

        logger.exception("Gemini API call failed: %s", e)

        return ""


# ---------------- LLM LATEX CONVERSION ---------------- #

def llm_convert_to_latex(ocr_text: str, image=None):

    cleaned = _clean_ocr_text(ocr_text)

    prompt_path = os.path.join(
        os.path.dirname(__file__), "../prompts/latex_prompt.txt"
    )

    try:
        prompt_template = open(prompt_path).read()
    except Exception:
        logger.exception("Prompt file not found")
        return cleaned

    prompt = prompt_template.replace("<<OCR_TEXT>>", cleaned)

    # -------- GEMINI -------- #

    gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key:

        out = _call_gemini_api(prompt, gemini_key)

        if out:
            return out.strip()

    # -------- OPENAI -------- #

    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key and _have_openai:

        try:

            client = OpenAI(api_key=openai_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.exception("OpenAI conversion failed: %s", e)

    # -------- FALLBACK -------- #

    return cleaned