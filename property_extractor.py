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

from PIL import Image
from pdf2image import convert_from_path
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

    # Optional: Poppler path if pdf2image needs it on Windows
    poppler_path: Optional[Path] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# ---- UTILS: FILE → IMAGES -------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}


def file_to_page_images(path: Path, max_pages: int, poppler_path: Optional[Path]) -> List[Image.Image]:
    """
    Load up to max_pages pages as PIL Images.
    - PDF: convert first N pages to images
    - Image: just load the single image (counts as 1 'page')
    - Other: returns empty list (we only handle PDFs and images here)
    """
    ext = path.suffix.lower()

    if ext in PDF_EXTS:
        # convert_from_path returns a list of PIL Images
        pages = convert_from_path(str(path), first_page=1, last_page=max_pages,
                                  poppler_path=str(poppler_path) if poppler_path else None)
        return pages[:max_pages]

    if ext in IMAGE_EXTS:
        try:
            img = Image.open(path)
            # Convert to RGB for consistency (some PDFs produce RGBA; some models prefer jpeg-like)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return [img]
        except Exception:
            return []

    return []


def pil_to_base64_jpeg(img: Image.Image) -> str:
    """Encode a PIL image to base64 JPEG (reasonable size/quality)."""
    buf = io.BytesIO()
    # Save with moderate quality to keep payload small but legible
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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

    # Prepare LLM client; if the CSV row specifies a different llm/model we will reconstruct as needed
    client_cache: Dict[str, LLMClient] = {}

    enriched_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        label = (row_dict.get("label") or "").strip()
        file_path = Path(row_dict.get("path", "")).expanduser()
        row_llm = (row_dict.get("llm") or settings.llm_name).strip().lower()
        row_model = (row_dict.get("model") or settings.llm_model).strip()

        # Get fields for this classification
        fields = mapping.get(label, [])
        if not fields:
            # No extraction mapping for this label—carry row through unchanged
            enriched_rows.append(row_dict)
            continue

        # Gather up to N page-images
        pages = []
        if file_path.exists():
            try:
                pages = file_to_page_images(file_path, settings.max_pages, settings.poppler_path)
            except Exception as e:
                # If conversion fails, we still pass an empty image list to the model (it may rely on prompt only)
                pages = []
        else:
            # file missing; keep row, no extraction
            enriched_rows.append(row_dict)
            continue

        # Build LLM prompt for this label/fields
        prompt = build_prompt(prompt_template, fields)

        # LLM client (use cache by llm_name|model)
        key = f"{row_llm}|{row_model}"
        if key not in client_cache:
            client_cache[key] = LLMClient(row_llm, settings.llm_api_key, row_model)
        client = client_cache[key]

        try:
            result = client.ask_with_images(prompt, pages)
            extracted = result.parsed_json if isinstance(result.parsed_json, dict) else {}
        except Exception as e:
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
