"""
Simple multi-LLM, multimodal property extractor (CLI)

Copyright (c) 2025 by Thomas J. Daley. All rights reserved.
See accompanying LICENSE file in repository root for details.

- Recursively walks a root directory, skipping folders that start with "."
- Converts each file to up to 5 PNG "pages" (PDF pages, images, or text rendered to image)
- Sends those images + a prompt (from file) to the chosen LLM
- Extracts fields based on a JSON mapping file
- Writes out an enriched CSV with original data + extracted fields

Usage:
  $ python -m venv venv
  $ source venv/bin/activate    # On Unix/macOS
  $ venv\\Scripts\\activate.bat   # On Windows
  $ pip install -r requirements.txt
  $ python classify_docs.py /path/to/root

Notes:
- For PDFs we use PyMuPDF (pure Python) to render images—no system deps.
- For image files we normalize to PNG bytes.
- For text-like files we render the first chunk of text onto a white PNG to preserve “layout” for the LLM.
- Keep it simple: one request per file, first 5 pages/images max.
"""

import base64
import csv
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from PIL import Image, ImageDraw, ImageFont
import fitz
import pandas as pd

# ---- CONFIG ---------------------------------------------------------------

class Settings(BaseSettings):
    # Which LLM to use: "openai", "anthropic", or "gemini"
    llm_name: str = Field(default="openai")

    # Your API key for the chosen LLM
    llm_api_key: str

    # The specific model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-latest", "gemini-1.5-flash")
    llm_model: str = Field(default="gpt-4o-mini")

    # Path to the previous program's output CSV (input for this program)
    input_csv: Path = Field(default=Path("classified_output.csv"))

    # Path to write the enriched CSV
    output_csv: Path = Field(default=Path("enriched_output.csv"))

    # Path to a prompt template file with placeholders {fields_list}
    prompt_file: Path = Field(default=Path("prompt_enrichment.txt"))

    # Path to JSON mapping: { "<classification>": ["field1", "field2", ...], ... }
    mapping_file: Path = Field(default=Path("mapping.json"))

    # How many pages/images to send per file (first N pages)
    max_pages: int = Field(default=5)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# ---- UTILS: FILE → IMAGES -------------------------------------------------

TEXT_EXTS = {".txt", ".md", ".eml", ".log", ".csv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
SETTINGS = Settings()
MAX_PAGES = SETTINGS.max_pages

def is_hidden_dir(path: Path) -> bool:
    return path.name.startswith(".")


def read_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def img_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def render_pdf_to_images(pdf_path: Path, max_pages: int = MAX_PAGES, zoom: float = 2.0) -> List[bytes]:
    """
    Render first `max_pages` pages of a PDF to PNG bytes using PyMuPDF.
    Zoom 2.0 ~ 144 DPI; adjust up/down if needed.

    Args:
        pdf_path: Path to the PDF file.
        max_pages: Maximum number of pages to render.
        zoom: Zoom factor for rendering.

    Returns:
        List of PNG byte strings, one per page.
    """
    images: List[bytes] = []
    with fitz.open(pdf_path.as_posix()) as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            page = doc.load_page(i)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    return images


def load_image_file(img_path: Path) -> List[bytes]:
    """
    Normalize any single image to PNG bytes (one page).
    """
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        return [img_to_png_bytes(im)]


def wrap_text_to_image(text: str, width_px: int = 1600, height_px: int = 2000, margin: int = 40, line_spacing: int = 6) -> bytes:
    """
    Draw text onto a white PNG. Uses a default Pillow font for simplicity.

    Args:
        text: The text to render.
        width_px: Width of the image in pixels.
        height_px: Height of the image in pixels.
        margin: Margin in pixels.
        line_spacing: Extra spacing between lines in pixels.

    Returns:
        PNG byte string of the rendered text image.
    """
    # Basic wrapping—greedy by words
    font = ImageFont.load_default()
    draw_img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(draw_img)

    max_text_width = width_px - 2 * margin
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_text_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    y = margin
    for line in lines:
        draw.text((margin, y), line, fill="black", font=font)
        y += font.getbbox(line)[3] - font.getbbox(line)[1] + line_spacing
        if y > height_px - margin:
            break

    return img_to_png_bytes(draw_img)


def load_text_as_images(path: Path) -> List[bytes]:
    """
    Read the first N chars and render as a single image.
    (We keep it to one image for simplicity.)
    """
    text = path.read_text(errors="ignore", encoding="utf-8")[:TEXT_CHARS_LIMIT]
    return [wrap_text_to_image(text)]


def file_to_images(path: Path) -> Optional[List[bytes]]:
    """
    Convert a file to up to 5 PNG images (bytes) depending on type.
    Returns None if the file type is unsupported.
    """
    ext = path.suffix.lower()
    try:
        if ext in PDF_EXTS:
            return render_pdf_to_images(path, MAX_PAGES)
        elif ext in IMAGE_EXTS:
            return load_image_file(path)
        elif ext in TEXT_EXTS:
            return load_text_as_images(path)
        else:
            # Unknown—try best effort: if it's tiny binary or unknown, skip.
            return None
    except Exception as e:
        print(f"[warn] Skipping {path}: {e}")
        return None


# =========== LLM CALLS ===========

def to_b64_images(png_bytes_list: List[bytes]) -> List[str]:
    return [base64.b64encode(b).decode("utf-8") for b in png_bytes_list[:MAX_PAGES]]


# ---- LLM ADAPTERS ---------------------------------------------------------

@dataclass
class LLMResult:
    raw_text: str
    parsed_json: Dict[str, Any]


class LLMClient:
    def __init__(self, name: str, api_key: str, model: str):
        self.name = name.lower()
        self.api_key = api_key
        self.model = model

        if self.name == "openai":
            from openai import OpenAI
            os.environ["OPENAI_API_KEY"] = api_key
            self.client = OpenAI()
        elif self.name == "anthropic":
            import anthropic
            os.environ["ANTHROPIC_API_KEY"] = api_key
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.name == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name=model)
        else:
            raise ValueError("Unsupported llm_name. Use 'openai', 'anthropic', or 'gemini'.")

    def _ensure_json(self, text: str) -> Dict[str, Any]:
        """
        Try to parse JSON from the model text.
        If the model wrapped it in code fences or added extra text,
        try to recover a JSON object.
        """
        text = text.strip()
        # Strip common code fence patterns
        if text.startswith("```"):
            # remove first fence line
            text = text.split("```", 2)
            if len(text) >= 2:
                text = text[1]
            else:
                text = text[0]
            # sometimes there's a language hint on the first line
            text = "\n".join(line for line in text.splitlines() if not line.lower().startswith(("json", "javascript")))
        # Attempt direct JSON load
        try:
            return json.loads(text)
        except Exception:
            # Last resort: find first '{' to last '}' substring
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    pass
        # If all fails, return empty dict
        return {}

    def ask_with_images(self, prompt: str, pil_images: List[Image.Image]) -> LLMResult:
        """
        Send a prompt with up to N images to the model.
        Expect a strictly-JSON response with the requested fields.
        """
        if self.name == "openai":
            # OpenAI Vision (responses.create with 'input_text' + 'input_image')
            image_items = []
            for img in pil_images:
                b64 = pil_to_base64_jpeg(img)
                image_items.append({"type": "input_image", "image_data": b64})

            content = [{"type": "input_text", "text": prompt}] + image_items

            resp = self.client.responses.create(
                model=self.model,
                input=content,
                # Modestly nudge the model toward concise JSON
                temperature=0.0,
            )
            text = resp.output_text

        elif self.name == "anthropic":
            # Anthropic Messages API with images (base64-jpeg)
            import anthropic

            media = []
            for img in pil_images:
                b64 = pil_to_base64_jpeg(img)
                media.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64
                    }
                })

            messages = [
                {"type": "text", "text": prompt},
                *media
            ]

            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                messages=[{"role": "user", "content": messages}]
            )
            # Concatenate all text blocks (usually one)
            text = "".join(block.text for block in resp.content if block.type == "text")

        elif self.name == "gemini":
            # Google Gemini Multimodal
            import google.generativeai as genai

            inputs = [prompt]
            for img in pil_images:
                # Gemini accepts PIL images directly
                inputs.append(img)

            resp = self.client.generate_content(inputs)
            text = resp.text or ""

        else:
            raise ValueError("Unsupported llm_name")

        parsed = self._ensure_json(text)
        return LLMResult(raw_text=text, parsed_json=parsed)

# ---- LLM HELPERS ----------------------------------------------------------
def to_b64_images(png_bytes_list: List[bytes]) -> List[str]:
    return [base64.b64encode(b).decode("utf-8") for b in png_bytes_list[:MAX_PAGES]]


def classify_with_openai(api_key: str, prompt: str, pngs: List[bytes], model: str = "gpt-4o-mini") -> str:
    """
    Minimal OpenAI image+text call using Chat Completions (gpt-4o-mini).

    Args:
        api_key: OpenAI API key.
        prompt: Text prompt to send.
        pngs: List of PNG byte strings.
    Returns:
        Classification label as a string.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    b64s = to_b64_images(pngs)
    content = [{"type": "text", "text": prompt}]
    for b64 in b64s:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful document classifier. Respond with a short, lowercase label like: bank_statement, credit_card_statement, email, text_message, social_media_message, or other."},
            {"role": "user", "content": content}
        ],
        temperature=0
    )
    return (resp.choices[0].message.content or "").strip()


def classify_with_anthropic(api_key: str, prompt: str, pngs: List[bytes], model: str = "claude-3-5-sonnet-latest") -> str:
    """
    Minimal Anthropic Claude 3.5 Sonnet image+text call.

    Args:
        api_key: Anthropic API key.
        prompt: Text prompt to send.
        pngs: List of PNG byte strings.
    Returns:
        Classification label as a string.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    parts = [{"type": "text", "text": prompt}]
    for b in pngs[:MAX_PAGES]:
        parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(b).decode("utf-8"),
            },
        })

    resp = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        system="You classify documents. Output a short, lowercase label only.",
        messages=[{"role": "user", "content": parts}],
    )
    return (resp.content[0].text if resp.content else "").strip()


def classify_with_gemini(api_key: str, prompt: str, pngs: List[bytes], model: str = "gemini-1.5-flash") -> str:
    """
    Minimal Google Gemini (1.5 Flash) image+text call.

    Args:
        api_key: Google Generative AI API key.
        prompt: Text prompt to send.
        pngs: List of PNG byte strings.
    Returns:
        Classification label as a string.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model)

    # Build a list of parts: prompt text + image blobs
    parts = [prompt]
    for b in pngs[:MAX_PAGES]:
        parts.append({"mime_type": "image/png", "data": b})

    resp = model.generate_content(parts)
    return (getattr(resp, "text", "") or "").strip()

def classify_images(llm_name: str, api_key: str, prompt: str, pngs: List[bytes], model: str) -> str:
    name = llm_name.strip().lower()
    if name == "openai":
        return classify_with_openai(api_key, prompt, pngs, model)
    if name == "anthropic":
        return classify_with_anthropic(api_key, prompt, pngs, model)
    if name == "gemini":
        return classify_with_gemini(api_key, prompt, pngs, model)
    raise ValueError(f"Unsupported LLM_NAME: {llm_name}")

# ---- PROMPT BUILDER -------------------------------------------------------

def load_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template_text: str, fields: List[str]) -> str:
    """
    The template should include '{fields_list}' placeholder.
    We'll also append a gentle instruction to return STRICT JSON.
    """
    fields_list = ", ".join(fields)
    prompt = template_text.format(fields_list=fields_list)

    strict_json_addendum = (
        "\n\nReturn ONLY valid JSON with these keys. "
        "Do not include any commentary, schema, code fences, or extra text—JSON object only."
    )
    return prompt + strict_json_addendum


# ---- MAIN PIPELINE --------------------------------------------------------

def main():
    settings = Settings()

    # Load mapping: classification -> list of field names
    mapping: Dict[str, List[str]] = json.loads(Path(settings.mapping_file).read_text(encoding="utf-8"))

    # Load prompt template
    prompt_template = load_text(settings.prompt_file)

    # Read input CSV
    df = pd.read_csv(settings.input_csv, dtype=str).fillna("")

    enriched_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        label = (row_dict.get("label") or "").strip()
        file_path = Path(row_dict.get("path", "")).expanduser()
        row_llm = (row_dict.get("llm") or settings.llm_name).strip().lower()
        row_model = (row_dict.get("model") or settings.llm_model).strip()
        print("Processing:", file_path, "Label:", label, "LLM:", row_llm, "Model:", row_model)

        # Get fields for this classification
        fields = mapping.get(label, [])
        if not fields:
            # No extraction mapping for this label—carry row through unchanged
            print(f"  No mapping for label '{label}'; skipping extraction.")
            enriched_rows.append(row_dict)
            continue

        # Gather up to N page-images
        images = []
        if file_path.exists():
            try:
                images = file_to_images(file_path) or []
            except Exception as e:
                print(f"  Error converting file to images: {e}")
                # If conversion fails, we still pass an empty image list to the model (it may rely on prompt only)
                images = []
        else:
            # file missing; keep row, no extraction
            print(f"  File not found: {file_path}; skipping extraction.")
            enriched_rows.append(row_dict)
            continue

        # Build LLM prompt for this label/fields
        prompt = build_prompt(prompt_template, fields)

        try:
            result = classify_images(
                llm_name=settings.llm_name,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                prompt=prompt,
                pngs=images
            )
            extracted = json.loads(result)
        except Exception as e:
            print(f"  Prompt: {prompt}")
            print(f"  Error during LLM extraction: {e}\n\n")
            extracted = {}

        # Merge extracted fields into row (only the mapped fields)
        for f in fields:
            row_dict[f] = extracted.get(f, "")

        enriched_rows.append(row_dict)

    # Union of columns: original + all extracted fields (from mapping)
    all_columns = list(df.columns)
    # add any new fields from mapping in stable order by label
    seen = set(all_columns)
    for lbl in mapping:
        for f in mapping[lbl]:
            if f not in seen:
                all_columns.append(f)
                seen.add(f)

    # Write output CSV
    output_path = Path(settings.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for r in enriched_rows:
            writer.writerow({k: r.get(k, "") for k in all_columns})

    print(f"Done. Wrote: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
