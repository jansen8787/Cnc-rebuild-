"""
stage25_titleblock/titleblock.py — Module 2: Stage 2.5
=======================================================
Title block parser. Separate from symbol parsing (architecture Q4 decision).

Purpose: extract structured metadata from the drawing title block
(part name, scale, units, tolerance class, material, etc.).

Strategy:
    1. Detect the title block region — it is typically in the bottom-right
       corner (~25% of drawing width, ~20% of drawing height).
    2. Run OCR on that region (or use pdf_text_layer annotations that fall
       inside the region).
    3. Match known title block field keywords against extracted text.
    4. Build TitleBlockInfo.

Architecture note (Q4): keeping this separate from Stage 2 (symbol_parser)
means a title-block OCR failure cannot corrupt dimension annotations, and vice
versa. The two stages share only the annotations list as input.

Scale string parsing:
    "1:2"  → px_per_mm = render_dpi / 25.4 * (1/2) = scale factor
    "2:1"  → larger-than-real drawing
    If missing → None; Stage 4 tries other anchors.

Public API:
    parse_title_block(annotations, width, height) -> TitleBlockInfo
    extract_scale(scale_raw) -> Optional[float]   # returns ratio (real/drawing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2types import TitleBlockInfo  # noqa: E402


# ---------------------------------------------------------------------------
# Title block region heuristic
# ---------------------------------------------------------------------------

# Fraction of image (from the right/bottom) that is the title block
_TB_RIGHT_FRAC: float = 0.50   # right 50% of width
_TB_BOTTOM_FRAC: float = 0.30  # bottom 30% of height


def _is_in_title_block_region(
    bbox: dict,
    img_width: int,
    img_height: int,
) -> bool:
    """Return True if the bbox centroid is in the expected title-block region."""
    cx = bbox["x"] + bbox.get("w", 0) / 2.0
    cy = bbox["y"] + bbox.get("h", 0) / 2.0
    in_right  = cx >= img_width  * (1.0 - _TB_RIGHT_FRAC)
    in_bottom = cy >= img_height * (1.0 - _TB_BOTTOM_FRAC)
    return in_right and in_bottom


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

_RE_SCALE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)"
)

_KNOWN_UNITS = {"mm", "millimeter", "millimetre", "in", "inch", "inches"}

def _norm_str(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    return s if s else None


# Keyword groups (case-insensitive) that identify a field
_FIELD_KEYWORDS: list[tuple[str, re.Pattern]] = [
    ("part_name",         re.compile(r"(?:TITLE|PART\s*NAME|BENENNUNG|TEILE?-?NAME)\s*[:\-]?\s*(.+)", re.I)),
    ("drawing_number",    re.compile(r"(?:DWG\s*NO|DRAWING\s*NO|ZEICHNUNGS-?NR|ZNR)\s*[:\-]?\s*(.+)", re.I)),
    ("revision",          re.compile(r"(?:REV(?:ISION)?|ÄENDERUNG)\s*[:\-]?\s*(.+)", re.I)),
    ("material",          re.compile(r"(?:MATERIAL|WERKSTOFF)\s*[:\-]?\s*(.+)", re.I)),
    ("scale_raw",         re.compile(r"(?:SCALE|MASSTAB|MASSSTAB|M)\s*[:\-]?\s*(.+)", re.I)),
    ("date",              re.compile(r"(?:DATE|DATUM)\s*[:\-]?\s*(.+)", re.I)),
    ("author",            re.compile(r"(?:DRAWN|GEZEICHNET|ERSTELLT)\s*[:\-]?\s*(.+)", re.I)),
    ("sheet",             re.compile(r"(?:SHEET|BLATT)\s*[:\-]?\s*(.+)", re.I)),
    ("tolerance_general", re.compile(r"(?:TOL(?:ERANCE)?|ALLGEM(?:EIN)?\.?\s*TOL|ATol)\s*[:\-]?\s*(.+)", re.I)),
]

# Standalone scale pattern (no keyword prefix, just "1:2")
_RE_BARE_SCALE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)$")

# Units hint patterns
_RE_UNITS_MM = re.compile(r"\bALL\s*DIM\w*\s+IN\s+MM\b|\bMILLIMET|\bMM\b", re.I)
_RE_UNITS_IN = re.compile(r"\bALL\s*DIM\w*\s+IN\s+INCH|\bINCH|\b\"\B", re.I)


def extract_scale(scale_raw: Optional[str]) -> Optional[float]:
    """
    Parse a scale string like "1:2" or "2:1" into a drawing-to-reality ratio.

    "1:2" → the drawing is drawn at half size → real/px ratio = 2.0
    "2:1" → the drawing is twice real size → ratio = 0.5

    Returns:
        ratio (float) = real_mm / drawing_mm, or None if unparseable.
    """
    if not scale_raw:
        return None
    text = scale_raw.replace(",", ".").strip()
    m = _RE_SCALE.search(text)
    if not m:
        return None
    try:
        drawing = float(m.group(1))
        real    = float(m.group(2))
    except (ValueError, TypeError):
        return None
    if drawing <= 0:
        return None
    return round(real / drawing, 6)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_title_block(
    annotations: list,
    width: int,
    height: int,
) -> TitleBlockInfo:
    """
    Extract title block metadata from text annotations.

    Scans all annotations. Prioritises those in the title-block region
    (bottom-right), but falls back to any annotation with a matching keyword.

    Args:
        annotations: List of TextAnnotation from Stage 2.
        width:       Image width in pixels (for region heuristic).
        height:      Image height in pixels.

    Returns:
        TitleBlockInfo with all fields set or None.
    """
    # Separate title-block-region annotations from the rest
    tb_anns = [a for a in annotations
               if _is_in_title_block_region(a.bbox, width, height)]
    all_anns = tb_anns + [a for a in annotations if a not in tb_anns]

    fields: dict = {
        "part_name": None,
        "drawing_number": None,
        "revision": None,
        "material": None,
        "scale_raw": None,
        "units_hint": None,
        "date": None,
        "author": None,
        "sheet": None,
        "tolerance_general": None,
    }
    extra: dict = {}

    # Also check if the annotation was already classified as title_block
    # by the symbol parser
    for ann in all_anns:
        text = ann.raw_text.strip()
        if not text:
            continue

        # Check symbol parser already identified a title-block field
        if ann.parsed.token_type == "title_block" and ann.parsed.qualifier:
            key_upper = ann.parsed.qualifier.upper().replace(" ", "_")
            value = text.split(":", 1)[-1].strip() if ":" in text else text

            if "MATERIAL" in key_upper and fields["material"] is None:
                fields["material"] = _norm_str(value)
            elif ("SCALE" in key_upper or "MASSSTAB" in key_upper) and fields["scale_raw"] is None:
                # value might be "1:2" or "SCALE: 1:2" — extract the ratio part
                m_sc = _RE_SCALE.search(value)
                if m_sc:
                    fields["scale_raw"] = f"{m_sc.group(1)}:{m_sc.group(2)}"
                else:
                    fields["scale_raw"] = _norm_str(value)
            elif "TITLE" in key_upper or "PART" in key_upper:
                if fields["part_name"] is None:
                    fields["part_name"] = _norm_str(value)
            elif "REV" in key_upper:
                if fields["revision"] is None:
                    fields["revision"] = _norm_str(value)
            elif "DWG" in key_upper or "DRAWING" in key_upper:
                if fields["drawing_number"] is None:
                    fields["drawing_number"] = _norm_str(value)
            else:
                extra[key_upper] = _norm_str(value)
            continue

        # Try all keyword regexes
        matched = False
        for field_name, pattern in _FIELD_KEYWORDS:
            m = pattern.match(text)
            if m and fields.get(field_name) is None:
                val = _norm_str(m.group(1))
                if val:
                    # Scale special handling
                    if field_name == "scale_raw":
                        m_sc = _RE_SCALE.search(val)
                        if m_sc:
                            val = f"{m_sc.group(1)}:{m_sc.group(2)}"
                    fields[field_name] = val
                matched = True
                break

        if not matched:
            # Standalone scale check
            m_bare = _RE_BARE_SCALE.match(text)
            if m_bare and fields["scale_raw"] is None:
                fields["scale_raw"] = text

        # Units hint
        if fields["units_hint"] is None:
            if _RE_UNITS_MM.search(text):
                fields["units_hint"] = "mm"
            elif _RE_UNITS_IN.search(text):
                fields["units_hint"] = "in"

    return TitleBlockInfo(
        part_name=fields["part_name"],
        drawing_number=fields["drawing_number"],
        revision=fields["revision"],
        material=fields["material"],
        scale_raw=fields["scale_raw"],
        units_hint=fields["units_hint"],
        date=fields["date"],
        author=fields["author"],
        sheet=fields["sheet"],
        tolerance_general=fields["tolerance_general"],
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    results: list = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition))
        marker = f"  ({detail})" if detail else ""
        print(f"  {PASS if condition else FAIL}  {name}{marker}")

    print("\n── Stage 2.5: Title Block self-tests ──\n")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2_ocr"))
    from symbol_parser import parse_symbol  # noqa: F401

    from m2types import TextAnnotation  # noqa: F811

    def make_ann(text: str, x: int, y: int) -> TextAnnotation:
        from stage2_ocr.symbol_parser import parse_symbol as ps
        return TextAnnotation(
            id=f"ann_{x}_{y}", page=1, raw_text=text,
            parsed=ps(text),
            bbox={"x": x, "y": y, "w": 100, "h": 15},
            ocr_confidence=0.95,
        )

    # Simulate 800×600 image; title block = right 50%, bottom 30%
    W, H = 800, 600
    # Title-block region: x >= 400, y >= 420

    anns = [
        make_ann("MATERIAL: AL6061", 410, 430),   # in TB
        make_ann("SCALE: 1:2",       410, 450),   # in TB
        make_ann("TITLE: Bracket",   410, 470),   # in TB
        make_ann("DWG NO: D-1234",   410, 490),   # in TB
        make_ann("Ø12",              100, 100),   # not in TB
    ]

    tb = parse_title_block(anns, W, H)

    check("material extracted",       tb.material == "AL6061",
          f"got {tb.material!r}")
    check("scale_raw extracted",      tb.scale_raw == "1:2",
          f"got {tb.scale_raw!r}")
    check("part_name extracted",      tb.part_name == "Bracket",
          f"got {tb.part_name!r}")
    check("drawing_number extracted", tb.drawing_number == "D-1234",
          f"got {tb.drawing_number!r}")

    # ── extract_scale
    check("1:2 → ratio 2.0",    extract_scale("1:2") == 2.0)
    check("2:1 → ratio 0.5",    extract_scale("2:1") == 0.5)
    check("1:1 → ratio 1.0",    extract_scale("1:1") == 1.0)
    check("1:5 → ratio 5.0",    extract_scale("1:5") == 5.0)
    check("None → None",        extract_scale(None) is None)
    check("garbage → None",     extract_scale("foo bar") is None)
    check("comma decimal 1,2:1", extract_scale("1,2:1") is not None)

    # ── units hint
    anns_mm = [make_ann("ALL DIMENSIONS IN MM", 410, 430)]
    tb_mm = parse_title_block(anns_mm, W, H)
    check("units_hint mm",   tb_mm.units_hint == "mm",
          f"got {tb_mm.units_hint!r}")

    anns_in = [make_ann("ALL DIM IN INCH", 410, 430)]
    tb_in = parse_title_block(anns_in, W, H)
    check("units_hint in",   tb_in.units_hint == "in",
          f"got {tb_in.units_hint!r}")

    # ── non-title-block annotation not used
    anns_non = [make_ann("Ø12", 50, 50)]   # not in TB region
    tb_non = parse_title_block(anns_non, W, H)
    # part_name should be None (Ø12 is not a title block field)
    check("Ø12 outside TB not used as part_name", tb_non.part_name is None)

    # ── empty annotations → all None
    tb_empty = parse_title_block([], W, H)
    check("empty anns → part_name None",   tb_empty.part_name is None)
    check("empty anns → scale_raw None",   tb_empty.scale_raw is None)
    check("empty anns → extra empty dict", tb_empty.extra == {})

    # ── _is_in_title_block_region
    check("centroid in TB region",
          _is_in_title_block_region({"x": 450, "y": 450, "w": 10, "h": 10}, 800, 600))
    check("centroid not in TB region (top-left)",
          not _is_in_title_block_region({"x": 10, "y": 10, "w": 10, "h": 10}, 800, 600))

    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n── Title Block tests: {passed}/{total} passed ──\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_tests())
