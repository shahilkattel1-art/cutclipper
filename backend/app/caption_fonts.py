"""Shared caption font catalog and safe font-file lookup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "fonts.json"


@dataclass(frozen=True, slots=True)
class CaptionFontSpec:
    name: str
    category: str
    scripts: tuple[str, ...]
    fallback: str
    files: Mapping[str, str]

    @property
    def css_family(self) -> str:
        return f"'{self.name}', {self.fallback}"


def _load_catalog() -> tuple[CaptionFontSpec, ...]:
    try:
        raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    specs: list[CaptionFontSpec] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        files = item.get("files")
        specs.append(
            CaptionFontSpec(
                name=name,
                category=str(item.get("category", "System")),
                scripts=tuple(str(script) for script in item.get("scripts", ["latin"])),
                fallback=str(item.get("fallback", "sans-serif")),
                files=MappingProxyType(
                    {
                        str(style): str(filename)
                        for style, filename in (files.items() if isinstance(files, dict) else ())
                        if isinstance(filename, str) and Path(filename).name == filename
                    }
                ),
            )
        )
    return tuple(specs)


CAPTION_FONT_SPECS = _load_catalog()
FONT_BY_NAME: dict[str, CaptionFontSpec] = {spec.name: spec for spec in CAPTION_FONT_SPECS}
CAPTION_FONT_NAMES: tuple[str, ...] = tuple(spec.name for spec in CAPTION_FONT_SPECS)
CAPTION_FONT_FILES: frozenset[str] = frozenset(
    filename for spec in CAPTION_FONT_SPECS for filename in spec.files.values()
)
CAPTION_FONT_STYLES: tuple[str, ...] = ("regular", "bold", "italic", "bold_italic")


def is_valid_font(name: str) -> bool:
    return name in FONT_BY_NAME


def font_spec(name: str) -> CaptionFontSpec | None:
    return FONT_BY_NAME.get(name)


def available_styles(name: str) -> tuple[str, ...]:
    spec = FONT_BY_NAME.get(name)
    if not spec:
        return ()
    styles = [style for style in CAPTION_FONT_STYLES if style in spec.files]
    # Browsers/fontconfig can synthesize bold and italic from a regular face.
    if spec.files.get("regular"):
        return CAPTION_FONT_STYLES
    return tuple(styles)


def safe_font_file(filename: str) -> Path | None:
    if filename not in CAPTION_FONT_FILES:
        return None
    path = (_CATALOG_PATH.parent / filename).resolve()
    try:
        path.relative_to(_CATALOG_PATH.parent.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def catalog_for_client() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "category": spec.category,
            "scripts": list(spec.scripts),
            "fallback": spec.fallback,
            "files": dict(spec.files),
        }
        for spec in CAPTION_FONT_SPECS
    ]


@dataclass(frozen=True, slots=True)
class CachedFontMetrics:
    win_total: int
    win_asc: int
    win_desc: int
    cap_height: int
    advances: dict[int, int]
    default_advance: int


@lru_cache(maxsize=128)
def get_font_metrics(
    font_name: str,
    *,
    bold: bool = True,
    italic: bool = False,
) -> CachedFontMetrics | None:
    """Load and cache FreeType / TrueType font advance metrics.

    In ASS (VSFilter/libass), FontSize is scaled based on WinAscent + WinDescent
    (equivalent to Windows GDI lfHeight). Tracking these exact advance widths
    guarantees that bounding boxes and word widths match the burned-in subtitle
    positions pixel-perfectly across all fonts and font sizes.
    """

    spec = font_spec(font_name)
    filename: str | None = None
    if spec:
        if bold and italic:
            filename = (
                spec.files.get("bold_italic")
                or spec.files.get("bold")
                or spec.files.get("regular")
            )
        elif bold:
            filename = spec.files.get("bold") or spec.files.get("regular")
        elif italic:
            filename = spec.files.get("italic") or spec.files.get("regular")
        else:
            filename = spec.files.get("regular") or spec.files.get("bold")

    path: Path | None = None
    if filename:
        path = safe_font_file(filename)

    if not path or not path.is_file():
        # Try matching font filename directly in the assets directory
        clean_name = font_name.replace(" ", "")
        for suffix in ("-Bold.ttf", "-Regular.ttf", ".ttf"):
            candidate = _CATALOG_PATH.parent / f"{clean_name}{suffix}"
            if candidate.is_file():
                path = candidate
                break

    if not path or not path.is_file():
        # Default fallback to Mukta or Inter
        path = safe_font_file("Mukta-Bold.ttf") or safe_font_file("Inter-Bold.ttf")

    if not path or not path.is_file():
        return None

    try:
        from fontTools.ttLib import TTFont

        font = TTFont(path)
        os2 = font["OS/2"]
        win_asc = int(os2.usWinAscent)
        win_desc = int(os2.usWinDescent)
        win_total = win_asc + win_desc
        if win_total <= 0:
            return None
        upem = int(font["head"].unitsPerEm)
        cap_h = int(getattr(os2, "sCapHeight", 0) or round(win_asc * 0.72))
        hmtx = font["hmtx"].metrics
        cmap = font.getBestCmap() or {}
        advances = {
            code: int(hmtx[g_name][0])
            for code, g_name in cmap.items()
            if g_name in hmtx
        }
        default_advance = round(upem * 0.5)
        return CachedFontMetrics(
            win_total=win_total,
            win_asc=win_asc,
            win_desc=win_desc,
            cap_height=cap_h,
            advances=advances,
            default_advance=default_advance,
        )
    except Exception:
        return None


def measure_text_width(
    text: str,
    font_size: float,
    font_name: str = "Inter",
    *,
    bold: bool = True,
    italic: bool = False,
    letter_spacing: float = 0.0,
) -> float:
    """Measure the exact rendered pixel width of text rendered via libass."""

    if not text:
        return 0.0

    metrics = get_font_metrics(font_name, bold=bold, italic=italic)
    if metrics is not None and metrics.win_total > 0:
        total_adv = sum(metrics.advances.get(ord(c), metrics.default_advance) for c in text)
        width = (total_adv / metrics.win_total) * font_size
    else:
        # Fallback ratio (~0.52 em per char) if font loading is unavailable
        width = len(text) * font_size * 0.52

    if letter_spacing != 0.0:
        width += len(text) * letter_spacing
    return max(0.0, width)

