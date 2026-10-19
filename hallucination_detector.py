"""Conservative quality checks for AI-generated image transcriptions."""

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


class HallucinationDetector:
    """Identify transcriptions that should receive human review."""

    def analyze_text(self, text: str, image_path: str) -> Dict[str, Any]:
        reasons: List[str] = []
        quality_metrics: Dict[str, float] = {"dpi": 0.0}
        width = height = 0

        try:
            with Image.open(Path(image_path)) as image:
                dpi = image.info.get("dpi", (0, 0))
                if isinstance(dpi, tuple):
                    quality_metrics["dpi"] = float(dpi[0] or 0)
                else:
                    quality_metrics["dpi"] = float(dpi or 0)
                width, height = image.size
        except (OSError, TypeError, ValueError) as exc:
            reasons.append(f"Could not inspect source image: {exc}")

        if not text.strip():
            reasons.append("The model returned no visible text.")
        if width and height and min(width, height) < 600:
            reasons.append("The source image has a short dimension below 600 pixels.")
        if 0 < quality_metrics["dpi"] < 150:
            reasons.append("The source image reports less than 150 DPI.")
        if "[ERROR:" in text:
            reasons.append("The transcription contains a processing error.")

        return {
            "needs_review": bool(reasons),
            "review_reasons": reasons,
            "quality_metrics": quality_metrics,
            "text_length": len(text),
        }

    def clean_hallucinations(self, text: str) -> str:
        """Leave uncertain text unchanged; review flags must not destroy evidence."""
        return text
