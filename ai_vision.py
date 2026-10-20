#!/usr/bin/env python3
"""
AI Vision-Based OCR Script
Uses OpenAI's GPT-4 Vision to extract text from images while preserving layout,
structure, and handling multiple languages. Provides accurate transcription
without interpretation or translation.
"""

import os
import base64
import json
import time
import argparse
import mimetypes
import sys
import tempfile
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from openai import OpenAI, OpenAIError
from hallucination_detector import HallucinationDetector

MAX_TRANSCRIPTION_CHARS = 20000
REGION_MAX_HEIGHT = 1800
LAYOUT_CLASSIFICATION_MAX_TOKENS = 300
REGION_OVERLAP = 24

# OpenAI API pricing (as of 2024 - verify current rates)
PRICING = {
    'gpt-4-vision-preview': {
        'input_tokens': 0.01 / 1000,      # $0.01 per 1K tokens
        'output_tokens': 0.03 / 1000,     # $0.03 per 1K tokens
    },
    'gpt-4o': {
        'input_tokens': 0.005 / 1000,     # $0.005 per 1K tokens  
        'output_tokens': 0.015 / 1000,    # $0.015 per 1K tokens
    }
}

class TokenTracker:
    """Track token usage and calculate costs for OpenAI API calls."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.model_usage = {}
    
    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add token usage for a specific model."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        if model not in self.model_usage:
            self.model_usage[model] = {'input': 0, 'output': 0, 'requests': 0}
        
        self.model_usage[model]['input'] += input_tokens
        self.model_usage[model]['output'] += output_tokens
        self.model_usage[model]['requests'] += 1
    
    def calculate_cost(self) -> float:
        """Calculate total cost based on token usage."""
        total_cost = 0.0
        
        for model, usage in self.model_usage.items():
            if model in PRICING:
                model_cost = (
                    usage['input'] * PRICING[model]['input_tokens'] +
                    usage['output'] * PRICING[model]['output_tokens']
                )
                total_cost += model_cost
        
        return total_cost
    
    def get_summary(self) -> Dict:
        """Get detailed usage and cost summary."""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_requests': self.total_requests,
            'total_cost': self.calculate_cost(),
            'model_breakdown': self.model_usage
        }

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 string for OpenAI API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def optimize_image_for_api(image_path: str, max_size: Tuple[int, int] = (2048, 2048)) -> str:
    """
    Optimize image size for API while maintaining quality.
    
    Args:
        image_path (str): Path to the original image
        max_size (Tuple[int, int]): Maximum dimensions (width, height)
        
    Returns:
        str: Path to optimized image (or original if no optimization needed)
    """
    try:
        with Image.open(image_path) as img:
            # Check if image needs resizing
            if img.size[0] <= max_size[0] and img.size[1] <= max_size[1]:
                return image_path


            # Calculate new size maintaining aspect ratio
            ratio = min(max_size[0] / img.size[0], max_size[1] / img.size[1])
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            
            # Resize and save optimized version
            optimized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            source_path = Path(image_path)
            optimized_path = str(
                source_path.with_name(f"{source_path.stem}_optimized{source_path.suffix}")
            )
            
            # Convert to RGB if necessary for JPEG
            if optimized_img.mode != 'RGB' and optimized_path.lower().endswith(('.jpg', '.jpeg')):
                optimized_img = optimized_img.convert('RGB')
            
            save_options: Dict[str, Any] = {}
            if optimized_img.format in {"JPEG", "WEBP"}:
                save_options.update(quality=95, optimize=True)
            optimized_img.save(optimized_path, **save_options)
            return optimized_path
            
    except Exception as e:
        print(f"Warning: Could not optimize image {image_path}: {e}")
        return image_path


def classify_document_layout(
    image_path: str,
    client: OpenAI,
    tracker: TokenTracker,
    model: str,
    max_image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Classify the page layout before choosing a transcription prompt."""
    optimized_path: Optional[str] = None
    try:
        optimized_path = optimize_image_for_api(image_path, max_image_size)
        base64_image = encode_image_to_base64(optimized_path)
        mime_type = mimetypes.guess_type(optimized_path)[0] or "image/jpeg"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Classify the visible document layout and, when a map is present, locate its regions. Return ONLY a JSON object:
{"layout_type": "paragraph", "table", "mixed", "map", or "mixed_map", "confidence": 0.0 to 1.0, "reason": "brief visual evidence", "regions": [{"type": "map" or "prose", "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}]}

Region coordinates must be normalized to 0.0-1.0 relative to the complete
image. For mixed_map, return non-overlapping regions that cover each visible
map and each nearby prose area. Include the map's complete drawing, labels,
and boundary context in its map region. Include marginal notes or headings
as prose regions when they are separate from the map. Sort regions by their
physical page position (top coordinate, then left coordinate). Do not return
regions for blank space. If a map is not present, return an empty regions list.

Use "table" only when most of the page is a visibly bounded grid, ledger, or
repeated row-and-column structure. Use "mixed" when a bounded table occupies
only part of the page and prose, headings, or certificates occupy another part.
Use "map" when the page is primarily a hand-drawn map, parcel diagram, route
diagram, or other spatial drawing with labels. Use "mixed_map" when a map or
diagram occupies only part of the page and prose occupies another part.
Use "paragraph" for prose, vertical lists, or multiple text columns that do not
form repeated rows with stable column positions. Do not call aligned text a table
merely because it has several columns. Base the decision on visible layout
boundaries, spatial relationships, and repeated rows, not document subject matter.""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=LAYOUT_CLASSIFICATION_MAX_TOKENS,
            temperature=0,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        if usage is not None:
            tracker.add_usage(
                model,
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
            )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The layout classifier returned an empty response.")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("The layout classifier returned an invalid JSON object.")
        layout_type = result.get("layout_type")
        if layout_type not in {"paragraph", "table", "mixed", "map", "mixed_map"}:
            # Accept the previous classifier shape if a model or cached response
            # still returns it, while defaulting conservatively.
            if result.get("has_map") is True:
                layout_type = "map"
            elif result.get("has_table") is True:
                layout_type = "table"
            else:
                layout_type = "paragraph"
        result["layout_type"] = layout_type
        result["has_table"] = layout_type in {"table", "mixed"}
        result["has_map"] = layout_type in {"map", "mixed_map"}
        result["confidence"] = float(result.get("confidence", 0.0))
        result["reason"] = str(result.get("reason", "")).strip()
        result["regions"] = _validated_layout_regions(result.get("regions"))
        return result
    except (OpenAIError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        print(f"Warning: Could not classify layout for {image_path}: {exc}")
        return {
            "has_table": False,
            "has_map": False,
            "layout_type": "paragraph",
            "confidence": 0.0,
            "reason": "Layout classification failed; used the standard transcription path.",
            "classification_error": str(exc),
            "regions": [],
        }
    finally:
        if (
            optimized_path
            and optimized_path != image_path
            and os.path.exists(optimized_path)
        ):
            os.remove(optimized_path)


def _validated_layout_regions(raw_regions: Any) -> List[Dict[str, Any]]:
    """Validate normalized layout regions returned by the vision model."""
    if not isinstance(raw_regions, list):
        return []

    regions: List[Dict[str, Any]] = []
    for raw_region in raw_regions:
        if not isinstance(raw_region, dict):
            continue
        region_type = raw_region.get("type")
        if region_type not in {"map", "prose"}:
            continue
        try:
            x = float(raw_region["x"])
            y = float(raw_region["y"])
            width = float(raw_region["width"])
            height = float(raw_region["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all(math.isfinite(value) for value in (x, y, width, height))
            or width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > 1
            or y + height > 1
        ):
            continue
        regions.append(
            {
                "type": region_type,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
    return sorted(regions, key=lambda region: (region["y"], region["x"]))


def remove_synthetic_table_placeholders(text: str) -> Tuple[str, bool]:
    """Remove the placeholder token from table output under the table contract."""
    if "[blank]" not in text:
        return text, False

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        cleaned = line.replace("[blank]", "").rstrip()
        if cleaned.strip("\t ").strip():
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines), True


def extract_text_with_ai_vision(
    image_path: str,
    client: OpenAI,
    tracker: TokenTracker,
    detector: HallucinationDetector,
    model: str = "gpt-4o",
    max_image_size: Tuple[int, int] = (2048, 2048),
    table_mode: bool = False,
    layout_mode: str = "paragraph",
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from image using OpenAI's Vision API with hallucination detection.
    
    Args:
        image_path (str): Path to the image file
        client (OpenAI): OpenAI client instance
        tracker (TokenTracker): Token usage tracker
        detector (HallucinationDetector): Instance of hallucination detector
        model (str): Model to use for vision processing
        
    Returns:
        Tuple[str, dict]: Tuple of (extracted_text, analysis_results)
    """
    try:
        # Optimize image for API
        optimized_path = optimize_image_for_api(image_path, max_image_size)
        
        # Encode image to base64
        base64_image = encode_image_to_base64(optimized_path)
        
        # Maps need a structured visual description; a paragraph prompt encourages
        # vision models to omit labels and relationships. Tables need a row/column
        # representation; a paragraph prompt encourages
        # vision models to flatten aligned values and then repeat them.
        if layout_mode in {"map", "mixed_map"}:
            prompt = f"""Transcribe this handwritten document with layout mode: {layout_mode}.
{"The page is primarily a hand-drawn map or spatial diagram." if layout_mode == "map" else "The page contains a hand-drawn map or spatial diagram plus non-map text."}

Use this map-specific workflow:
1. Read the page in physical reading order from top to bottom. When content is
   side by side, read from left to right. Keep the transcription order faithful
   to where content appears on the page.
2. Insert a clearly marked section exactly as "[HAND-DRAWN MAP]" at the map's
   position in that reading order, and end it with
   "[END HAND-DRAWN MAP]".
   For example, if the map is above a paragraph, the map section must come
   before that paragraph in the transcription. Do not move the map description
   to the end merely because it is a description.
3. Transcribe prose, headings, dates, and marginal notes where they occur
   relative to the map; do not summarize or reorder them.
4. Within the map section, record all readable map labels, parcel or tract names,
   adjoining owners, roads, waterways, buildings, measurements, bearings,
   acreage, orientation marks, arrows, corrections, and parcel numbers.
5. Describe the map's visible layout and relationships in concise text. State
   where features are located relative to one another (for example, north,
   south, left, right, adjoining, or central) without treating the drawing as
   geographically accurate or to scale.
6. Preserve the original spelling, abbreviations, punctuation, and numbers.
7. Distinguish visible text from interpretation. Use "[illegible]" when no
   characters can be read, "[?]" only for a visibly uncertain character, and
   "[possibly: ...]" for an uncertain interpretation.
8. Record crossed-out text, insertions, arrows, and boundary lines when visible.
9. Do not silently redraw, rotate, normalize, expand abbreviations, or correct
   the map. Do not invent labels, measurements, orientation, or relationships.
10. If a spatial relationship is unclear, say "[relationship uncertain]" rather
   than guessing.
11. Transcribe each visible item only once and stop at the last visible item.

Return ONLY the transcription and map description. Do not include Markdown,
commentary, or a summary outside the marked map section."""
        elif layout_mode in {"table", "mixed"} or table_mode:
            prompt = f"""Transcribe this handwritten document with layout mode: {layout_mode}.
{"A bounded table occupies most of the page." if layout_mode == "table" else "The page contains a bounded table plus non-table text."}

Rules:
1. Read from top to bottom. Within a real table, read each visible row from left to right.
2. Within a real table, use TAB characters only between cells that visibly belong to
   that row. Do not create a rectangular grid.
3. Preserve every visible cell exactly, including spelling, abbreviations, punctuation,
   names, and numbers.
4. Never output "[blank]" or any placeholder for empty space. If a cell is empty, leave
   that position empty between tabs only when the surrounding row and column boundaries
   are visibly established. Otherwise omit the empty cell.
5. Do not add a row merely because a column continues. Stop at the last visible row.
6. In non-table areas, use ordinary line breaks and prose layout; do not add tabs or
   placeholders to make prose fit table columns.
7. Never invent a value to fill a blank or align a damaged row.
8. Transcribe a row only once. Never repeat a row, cell, or sequence to fill output.
9. If a value is visibly uncertain, mark only the uncertain character with [?]; use
   [illegible] only when no character can be read.
10. Stop after the last visible text. Do not continue a pattern or restart at a new column.

Return ONLY the transcription. Do not include Markdown table syntax, commentary, or a
summary."""
        else:
            prompt = """Extract ALL text from this image exactly as it appears. Follow these strict guidelines:

1. Transcribe every piece of visible text, preserving the original layout and structure.
2. Preserve line breaks, paragraph breaks, and spatial relationships exactly as shown.
3. Preserve spelling exactly, including unusual, historical, or incorrect spelling.
4. Avoid silently expanding abbreviations; transcribe abbreviations exactly as written.
5. If a character or word is uncertain but partially visible, transcribe the visible text and mark the uncertain character with [?].
6. Distinguish uncertainty from illegibility: use [?] for a visible but uncertain character, and use [illegible] only when the character or text cannot be read at all.
7. Avoid adding words that are not visible in the image, even when they seem likely from context.
8. Preserve marginal notes, headings, titles, dates, numbers, and all other visible text.
9. If text is in a non-Latin script (Arabic, Chinese, Tamil, etc.), transcribe it exactly as written without translation.
10. Preserve the spatial layout using appropriate spacing and line breaks.
11. Identify multiple columns or sections and preserve their order and structure.
12. Do not add commentary, interpretation, corrections, or explanations to the transcription.
13. Do not repeat any word, line, paragraph, section, or marginal note.
14. Transcribe each visible line only once.
15. If the same text appears twice in the image, transcribe it twice only if it
    is visibly present twice.
16. Do not continue a previous line or paragraph unless that continuation is
    visible in the image.
17. If you lose your place, stop and mark the uncertain location with [?]
    rather than restarting or repeating text.

Return ONLY the transcribed text with no additional commentary."""

        # Make API call
        mime_type = mimetypes.guess_type(optimized_path)[0] or "image/jpeg"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=3000,
            temperature=0  # Use 0 for consistent, deterministic transcription
        )
        
        # Track token usage
        usage = response.usage
        if usage is not None:
            tracker.add_usage(
                model,
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
            )
        
        # Get the transcribed text
        content: Optional[str] = response.choices[0].message.content
        if not content:
            raise ValueError("The vision model returned an empty transcription.")
        transcribed_text = content.strip()
        placeholder_removed = False
        if layout_mode in {"table", "mixed"} or table_mode:
            transcribed_text, placeholder_removed = remove_synthetic_table_placeholders(
                transcribed_text
            )
        if len(transcribed_text) > MAX_TRANSCRIPTION_CHARS:
            raise ValueError(
                f"The transcription exceeded the {MAX_TRANSCRIPTION_CHARS:,}-character safeguard."
            )
        
        # Analyze for hallucinations
        analysis_results = detector.analyze_text(transcribed_text, image_path)
        if placeholder_removed:
            analysis_results["needs_review"] = True
            analysis_results["review_reasons"] = list(
                dict.fromkeys(
                    [
                        "Removed synthetic [blank] placeholders from table-mode output; verify empty cells against the image.",
                        *analysis_results.get("review_reasons", []),
                    ]
                )
            )
        
        # Clean up optimized image if it was created
        if optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)
        
        return transcribed_text, analysis_results
        
    except Exception as e:
        error_msg = f"[ERROR: Could not process image - {str(e)}]"
        print(f"Error processing {image_path}: {str(e)}")
        # Clean up optimized image on error
        if 'optimized_path' in locals() and optimized_path != image_path and os.path.exists(optimized_path):
            os.remove(optimized_path)
        return error_msg, {
            "needs_review": True,
            "review_reasons": [f"Processing error: {str(e)}"],
            "quality_metrics": {"dpi": 0.0},
            "text_length": 0
        }

def transcribe_image(
    image_path: str,
    client: OpenAI,
    tracker: TokenTracker,
    detector: HallucinationDetector,
    model: str,
    max_image_size: Tuple[int, int],
) -> Tuple[str, Dict[str, Any]]:
    """Transcribe a page using ordered map/prose regions when available."""
    layout = classify_document_layout(
        image_path, client, tracker, model, max_image_size
    )
    table_mode = bool(layout.get("has_table"))
    map_mode = bool(layout.get("has_map"))
    layout["authoritative_source"] = image_path
    with Image.open(image_path) as image:
        width, height = image.size
        detected_regions = layout.get("regions", [])
        can_use_layout_regions = (
            map_mode
            and isinstance(detected_regions, list)
            and any(region["type"] == "map" for region in detected_regions)
            and (
                layout.get("layout_type") != "mixed_map"
                or any(region["type"] == "prose" for region in detected_regions)
            )
        )
        if can_use_layout_regions:
            try:
                return _transcribe_layout_regions(
                    image,
                    image_path,
                    detected_regions,
                    client,
                    tracker,
                    detector,
                    model,
                    max_image_size,
                    table_mode,
                    layout,
                )
            except (OSError, ValueError) as exc:
                layout["region_error"] = str(exc)

        # Preserve the original whole-page path if layout analysis did not
        # return a usable map region. This is explicitly flagged for review.
        region_count = max(1, math.ceil(height / REGION_MAX_HEIGHT))
        if region_count == 1:
            text, analysis = extract_text_with_ai_vision(
                image_path,
                client,
                tracker,
                detector,
                model,
                max_image_size,
                table_mode,
                layout.get("layout_type", "paragraph"),
            )
            analysis["layout"] = layout
            if map_mode:
                analysis["needs_review"] = True
                analysis["review_reasons"] = list(
                    dict.fromkeys(
                        [
                            "Map region detection failed; the complete original image was transcribed as a fallback. Verify ordering and map boundaries.",
                            *(
                                [f"Region transcription failed: {layout['region_error']}"]
                                if layout.get("region_error")
                                else []
                            ),
                            *analysis.get("review_reasons", []),
                        ]
                    )
                )
            elif table_mode:
                analysis["needs_review"] = True
                analysis["review_reasons"] = list(
                    dict.fromkeys(
                        [
                            "Layout classifier identified a table or mixed layout; verify row and column alignment.",
                            *analysis.get("review_reasons", []),
                        ]
                    )
                )
            elif layout.get("classification_error"):
                analysis["needs_review"] = True
                analysis["review_reasons"] = list(
                    dict.fromkeys(
                        [
                            str(layout["reason"]),
                            *analysis.get("review_reasons", []),
                        ]
                    )
                )
            return text, analysis

        texts: List[str] = []
        region_reasons: List[str] = []
        for region_index in range(region_count):
            top = round(region_index * height / region_count)
            bottom = round((region_index + 1) * height / region_count)
            with tempfile.NamedTemporaryFile(
                suffix=Path(image_path).suffix, delete=False
            ) as region_file:
                region_path = region_file.name
            try:
                image.crop((0, top, width, bottom)).save(region_path)
                region_text, region_analysis = extract_text_with_ai_vision(
                    region_path,
                    client,
                    tracker,
                    detector,
                    model,
                    max_image_size,
                    table_mode,
                    layout.get("layout_type", "paragraph"),
                )
                texts.append(
                    f"[REGION {region_index + 1} OF {region_count}]\n{region_text}"
                )
                region_reasons.extend(region_analysis.get("review_reasons", []))
            finally:
                if os.path.exists(region_path):
                    os.remove(region_path)

        combined_text = "\n\n".join(texts)
        repetition_findings = detector.detect_repetition(combined_text)
        reasons = [
            f"Page was transcribed in {region_count} horizontal regions.",
            *region_reasons,
            *repetition_findings,
        ]
        if table_mode:
            reasons.insert(
                0,
                "Layout classifier identified a table or mixed layout; review region joins and row alignment.",
            )
        return combined_text, {
            "needs_review": True,
            "review_reasons": list(dict.fromkeys(reasons)),
            "repetition_findings": repetition_findings,
            "quality_metrics": {"dpi": 0.0},
            "text_length": len(combined_text),
            "regions": region_count,
            "layout": layout,
        }


def _transcribe_layout_regions(
    image: Image.Image,
    image_path: str,
    regions: List[Dict[str, Any]],
    client: OpenAI,
    tracker: TokenTracker,
    detector: HallucinationDetector,
    model: str,
    max_image_size: Tuple[int, int],
    table_mode: bool,
    layout: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Crop validated layout regions, transcribe them, and combine page order."""
    texts: List[str] = []
    review_reasons: List[str] = []
    for region in regions:
        left = max(0, round(region["x"] * image.width) - REGION_OVERLAP)
        top = max(0, round(region["y"] * image.height) - REGION_OVERLAP)
        right = min(
            image.width,
            round((region["x"] + region["width"]) * image.width) + REGION_OVERLAP,
        )
        bottom = min(
            image.height,
            round((region["y"] + region["height"]) * image.height) + REGION_OVERLAP,
        )
        if right <= left or bottom <= top:
            continue

        with tempfile.NamedTemporaryFile(
            suffix=Path(image_path).suffix, delete=False
        ) as region_file:
            region_path = region_file.name
        try:
            image.crop((left, top, right, bottom)).save(region_path)
            region_text, region_analysis = extract_text_with_ai_vision(
                region_path,
                client,
                tracker,
                detector,
                model,
                max_image_size,
                table_mode=table_mode and region["type"] == "prose",
                layout_mode="map" if region["type"] == "map" else "paragraph",
            )
            texts.append(region_text)
            review_reasons.extend(region_analysis.get("review_reasons", []))
        finally:
            if os.path.exists(region_path):
                os.remove(region_path)

    if not texts:
        raise ValueError("No usable layout regions remained after cropping.")

    combined_text = "\n\n".join(text.strip() for text in texts if text.strip())
    repetition_findings = detector.detect_repetition(combined_text)
    return combined_text, {
        "needs_review": True,
        "review_reasons": list(
            dict.fromkeys(
                [
                    "Map and prose were transcribed from detected regions in physical page order; verify crop boundaries and overlap against the complete original image.",
                    *review_reasons,
                    *repetition_findings,
                ]
            )
        ),
        "repetition_findings": repetition_findings,
        "quality_metrics": {"dpi": 0.0},
        "text_length": len(combined_text),
        "regions": len(texts),
        "layout_regions": regions,
        "layout": layout,
    }


def verify_transcription_with_ai_vision(
    image_path: str,
    transcribed_text: str,
    client: OpenAI,
    tracker: TokenTracker,
    model: str,
    max_image_size: Tuple[int, int] = (2048, 2048),
) -> str:
    """Compare a draft transcription with its source image and report concerns."""
    optimized_path: Optional[str] = None
    try:
        optimized_path = optimize_image_for_api(image_path, max_image_size)
        base64_image = encode_image_to_base64(optimized_path)
        map_instructions = ""
        if "[HAND-DRAWN MAP]" in transcribed_text:
            map_instructions = """
This draft contains a hand-drawn map section. Also check that map labels,
measurements, orientation marks, boundary relationships, and uncertainty
markers are supported by the image. Do not treat a prose description as a
replacement for the original drawing.
"""
        prompt = f"""Review the image against the draft transcription below.
Flag possible omissions, additions, repeated passages, or ordering problems.
Pay special attention to names, dates, numbers, abbreviations, marginal notes,
headings, multiple columns or sections.
{map_instructions}

Do not rewrite, correct, modernize, or silently change the draft. Do not say
what the historical text "should" say. Report only a possible reading or
evidence-based concern, quote the relevant draft text when possible, and state
that human review is required. If you find no obvious discrepancy, say so.

Draft transcription:
--- BEGIN DRAFT ---
{transcribed_text}
--- END DRAFT ---

Return a concise verification report with:
1. Verification status: NO OBVIOUS DISCREPANCIES or REVIEW REQUIRED
2. Numbered potential issues, if any
3. A reminder that the original image remains authoritative."""

        mime_type = mimetypes.guess_type(optimized_path)[0] or "image/jpeg"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=3000,
            temperature=0,
        )

        usage = response.usage
        if usage is not None:
            tracker.add_usage(
                model,
                usage.prompt_tokens or 0,
                usage.completion_tokens or 0,
            )

        content: Optional[str] = response.choices[0].message.content
        if not content:
            raise ValueError("The verification model returned an empty report.")
        return content.strip()
    except Exception as exc:
        print(f"Error verifying {image_path}: {exc}")
        return f"[ERROR: Could not verify transcription - {exc}]"
    finally:
        if (
            optimized_path
            and optimized_path != image_path
            and os.path.exists(optimized_path)
        ):
            os.remove(optimized_path)

def save_transcription(
    image_path: str,
    transcribed_text: str,
    output_dir: str,
    analysis_results: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save transcribed text to a file with hallucination analysis.
    
    Args:
        image_path (str): Path to source image
        transcribed_text (str): Extracted text
        output_dir (str): Directory to save the transcription
        analysis_results (dict): Optional results from hallucination analysis
        
    Returns:
        str: Path to saved file or None if error occurs
    """
    try:
        image_name = Path(image_path).name
        base_name = Path(image_path).stem
        output_file = os.path.join(
            output_dir, f"{base_name}_raw_transcription.txt"
        )
        
        # Save transcription with metadata
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Source Image: {image_name}\n")
            f.write(f"Transcription Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Add hallucination analysis if available
            if analysis_results:
                f.write("\nQuality Analysis:\n")
                f.write("-" * 40 + "\n")
                dpi = analysis_results.get("quality_metrics", {}).get("dpi")
                if dpi is not None:
                    f.write(f"Image DPI: {dpi:.1f}\n")
                layout = analysis_results.get("layout")
                if isinstance(layout, dict):
                    layout_type = layout.get(
                        "layout_type",
                        "table" if layout.get("has_table") else "paragraph",
                    )
                    confidence = float(layout.get("confidence", 0.0))
                    f.write(
                        f"Detected layout: {layout_type} "
                        f"(confidence {confidence:.2f})\n"
                    )
                    if layout.get("reason"):
                        f.write(f"Layout evidence: {layout['reason']}\n")
                if analysis_results['needs_review']:
                    f.write("Review Flags:\n")
                    for reason in analysis_results['review_reasons']:
                        f.write(f"- {reason}\n")
                f.write("\n")
                
            f.write("=" * 60 + "\n\n")
            f.write(transcribed_text)
            
        return output_file
        
    except Exception as e:
        print(f"Error saving transcription: {e}")
        return None

def save_verification_report(
    image_path: str,
    verification_report: str,
    output_dir: str,
) -> Optional[str]:
    """Save the independent verification report beside its transcription."""
    try:
        output_file = os.path.join(
            output_dir,
            f"{Path(image_path).stem}_verification.txt",
        )
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(f"Source Image: {Path(image_path).name}\n")
            file.write(f"Verification Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 60 + "\n\n")
            file.write(verification_report)
        return output_file
    except OSError as exc:
        print(f"Error saving verification report: {exc}")
        return None

def save_usage_summary(tracker: TokenTracker, output_dir: str, image_files: List[str]) -> None:
    """
    Save detailed usage and cost summary.
    
    Args:
        tracker (TokenTracker): Token usage tracker
        output_dir (str): Directory to save the summary
        image_files (List[str]): List of processed image files
    """
    summary = tracker.get_summary()
    
    # Create summaries subdirectory
    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Save as JSON
    json_file = os.path.join(summaries_dir, "usage_summary.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processing_summary': summary,
            'images_processed': [str(Path(img).name) for img in image_files],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'estimated_cost_usd': summary['total_cost']
        }, f, indent=2)
    
    # Save as readable text
    text_file = os.path.join(summaries_dir, "usage_summary.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("AI Vision OCR - Usage and Cost Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Processed: {len(image_files)}\n")
        f.write(f"Total API Requests: {summary['total_requests']}\n\n")
        
        f.write("Token Usage:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Input Tokens:  {summary['total_input_tokens']:,}\n")
        f.write(f"Output Tokens: {summary['total_output_tokens']:,}\n")
        f.write(f"Total Tokens:  {summary['total_tokens']:,}\n\n")
        
        f.write(f"Estimated Cost: ${summary['total_cost']:.4f} USD\n\n")
        
        if summary['model_breakdown']:
            f.write("Model Breakdown:\n")
            f.write("-" * 16 + "\n")
            for model, usage in summary['model_breakdown'].items():
                model_cost = 0
                if model in PRICING:
                    model_cost = (
                        usage['input'] * PRICING[model]['input_tokens'] +
                        usage['output'] * PRICING[model]['output_tokens']
                    )
                f.write(f"{model}:\n")
                f.write(f"  Requests: {usage['requests']}\n")
                f.write(f"  Input tokens: {usage['input']:,}\n")
                f.write(f"  Output tokens: {usage['output']:,}\n")
                f.write(f"  Cost: ${model_cost:.4f}\n\n")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a generic OCR run."""
    parser = argparse.ArgumentParser(
        description="Transcribe handwritten images with an OpenAI vision model."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images in subdirectories of --input-dir.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI vision model (default: OPENAI_MODEL or gpt-4o)",
    )
    parser.add_argument(
        "--verification-model",
        default=os.getenv("OPENAI_VERIFICATION_MODEL"),
        help="Model for the verification pass (default: same as --model)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ocr_results_ai_vision"),
        help="Directory for transcriptions and summaries (default: ocr_results_ai_vision)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between image requests (default: 1.0)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2048,
        help="Maximum width or height sent to the API (default: 2048)",
    )
    return parser.parse_args()

def main() -> int:
    """Main function to process all images with AI vision."""
    args = parse_args()

    if args.max_image_size <= 0:
        print("Error: --max-image-size must be greater than zero.")
        return 2
    if args.delay < 0:
        print("Error: --delay cannot be negative.")
        return 2

    # Initialize hallucination detector
    detector = HallucinationDetector()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return 2
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()
    verification_model = args.verification_model or args.model
    
    # Define paths
    base_dir = Path(__file__).parent
    base_output_dir = args.output_dir
    if not base_output_dir.is_absolute():
        base_output_dir = base_dir / base_output_dir
    
    print("Starting AI Vision OCR processing...")
    print(f"Base directory: {base_dir}")
    print(f"Base output directory: {base_output_dir}")
    
    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return 2
    processing_jobs = [(input_dir.name or "input", input_dir)]
    
    if not processing_jobs:
        print("No input directories with images found.")
        return 0
    
    print(f"Found {len(processing_jobs)} input director{'y' if len(processing_jobs) == 1 else 'ies'} to process:")
    for folder_name, images_dir in processing_jobs:
        print(f"  - {folder_name}: {images_dir}")
    print()
    
    # Process the selected input directory.
    all_results = {}
    total_start_time = time.time()
    
    for folder_index, (folder_name, images_dir) in enumerate(processing_jobs, 1):

        if not images_dir.exists() or not images_dir.is_dir():
            print(f"  Images directory not found: {images_dir}")
            print()
            continue

        output_dir = base_output_dir
        
        print(f"[{folder_index}/{len(processing_jobs)}] Processing {folder_name}...")
        print(f"  Images directory: {images_dir}")
        print(f"  Output directory: {output_dir}")
        
        # Get all image files for this folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}
        image_iterator = images_dir.rglob("*") if args.recursive else images_dir.iterdir()
        image_files = sorted(
            (
                f for f in image_iterator
                if f.is_file() and f.suffix.lower() in image_extensions
            ),
            key=lambda path: path.name.lower(),
        )
        
        if not image_files:
            print(f"  No image files found in {images_dir}")
            print()
            continue
        
        print(f"  Found {len(image_files)} image files:")
        for img_file in image_files:
            print(f"    - {img_file.name}")
        print()
        
        # Process each image in this folder
        folder_results = []
        
        for i, img_file in enumerate(image_files, 1):
            print(f"  [{i}/{len(image_files)}] Processing: {img_file.name}")
            
            # Extract text using AI vision with hallucination detection
            transcribed_text, analysis_results = transcribe_image(
                str(img_file),
                client,
                tracker,
                detector,
                model=args.model,
                max_image_size=(args.max_image_size, args.max_image_size),
            )

            verification_report: Optional[str] = None
            if not transcribed_text.startswith("[ERROR:"):
                print("    Running verification pass...")
                verification_report = verify_transcription_with_ai_vision(
                    str(img_file),
                    transcribed_text,
                    client,
                    tracker,
                    model=verification_model,
                    max_image_size=(args.max_image_size, args.max_image_size),
                )

            # Preserve flagged text and leave the review decision to a person.
            if analysis_results['needs_review']:
                print("  ⚠️ Quality issues detected:")
                for reason in analysis_results['review_reasons']:
                    print(f"    - {reason}")
                print("  Preserving text and marking it for review...")
                transcribed_text = detector.clean_hallucinations(transcribed_text)
            
            # Preserve subdirectory structure during recursive runs.
            image_output_dir = output_dir
            if args.recursive:
                relative_parent = img_file.parent.relative_to(input_dir)
                image_output_dir = output_dir / relative_parent
            image_output_dir.mkdir(parents=True, exist_ok=True)

            # Save transcription
            output_file = save_transcription(
                str(img_file),
                transcribed_text,
                str(image_output_dir),
                analysis_results,
            )
            verification_file = None
            if verification_report is not None:
                verification_file = save_verification_report(
                    str(img_file),
                    verification_report,
                    str(image_output_dir),
                )
            
            if output_file:
                print(f"    Saved to: {Path(output_file).name}")
                if verification_file:
                    print(f"    Verification saved to: {Path(verification_file).name}")
                print(f"    Text length: {len(transcribed_text)} characters")
            else:
                print("    Error: Could not save transcription")
            
            # Show preview of transcribed text
            if transcribed_text and not transcribed_text.startswith('[ERROR'):
                preview = transcribed_text[:150].replace('\n', ' ').strip()
                print(f"    Preview: {preview}{'...' if len(transcribed_text) > 150 else ''}")
            
            # Only append results if we successfully saved the file
            if output_file:
                folder_results.append({
                    'image': img_file.name,
                    'text': transcribed_text,
                    'output_file': str(output_file),
                    'verification_file': str(verification_file) if verification_file else None,
                    'verification_report': verification_report,
                    'analysis': analysis_results
                })
            
            # Small delay to respect API rate limits
            if i < len(image_files) and args.delay:
                time.sleep(args.delay)
        
        # Store results for this folder
        all_results[folder_name] = {
            'folder_path': images_dir,
            'images_processed': len(folder_results),
            'results': folder_results
        }
        
        print(f"  Completed {folder_name}: {len(folder_results)} images processed")
        print()
    
    # Calculate and display final summary
    end_time = time.time()
    processing_time = end_time - total_start_time
    summary = tracker.get_summary()
    
    # Calculate totals
    total_images = sum(data['images_processed'] for data in all_results.values())
    
    print(f"\n" + "=" * 70)
    print("AI Vision OCR Processing Complete!")
    print("=" * 70)
    print(f"Input directories processed: {len(all_results)}")
    print(f"Total images processed: {total_images}")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Total API requests: {summary['total_requests']}")
    print(f"Total tokens used: {summary['total_tokens']:,}")
    print(f"  - Input tokens: {summary['total_input_tokens']:,}")
    print(f"  - Output tokens: {summary['total_output_tokens']:,}")
    print(f"Estimated cost: ${summary['total_cost']:.4f} USD")
    print(f"\nResults saved in: {base_output_dir}")
    
    # Show folder breakdown
    print("\nFolder Processing Summary:")
    print("-" * 30)
    for folder_name, data in all_results.items():
        print(f"{folder_name}: {data['images_processed']} images processed")
    
    # Save detailed usage summary for each folder and overall
    all_image_files = []
    for folder_data in all_results.values():
        for result in folder_data['results']:
            all_image_files.append(result['image'])
    
    save_usage_summary(tracker, str(base_output_dir), all_image_files)
    print(f"Usage summary saved to: {base_output_dir / 'summaries' / 'usage_summary.txt'} and {base_output_dir / 'summaries' / 'usage_summary.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())