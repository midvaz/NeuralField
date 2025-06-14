"""Desktop interface for Agriculture‑Vision field analysis (Flet ≥ 0.28.3).

Features (desktop & web):
- Pick an image via file dialog.
- Show original photo side‑by‑side with ML‑processed segmentation overlay.
- Display detected Agriculture‑Vision classes, per‑class agronomic recommendations and optional GPS coordinates.

💡 Replace the stub `analyze_image()` with your own pipeline returning:
```python
classes: List[str]                           # detected class names
overlay_path: pathlib.Path                  # PNG/JPG with segmentation overlay
latlon: Optional[Tuple[float, float]]       # (lat, lon) or None if unavailable
```
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import flet as ft
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------------
# Agriculture‑Vision label → recommendation mapping
# -----------------------------------------------------------------------------
RECOMMENDATIONS = {
  "bare_soil": "Рассмотрите посев покровных культур или отрегулируйте норму высева для увеличения покрова почвы.",
  "cloud_shadow": "Изображение затенено облаком – выполните съемку при ясных условиях.",
  "double_plant": "Обнаружено двойное высеивание – перенастройте сеялку и контролируйте густоту посева.",
  "drydown": "Признаки дефицита влаги – оцените график ирригации или используйте засухоустойчивые гибриды.",
  "endrow": "Перекрытие на границах поля – уточните траекторию оборудования или настройте автопилот по границам.",
  "nutrient_deficiency": "Вероятен дефицит питательных веществ – проведите анализ почвы и внесите сбалансированные удобрения.",
  "planter_skip": "Обнаружены пропуски при высеве – проверьте высевающие секции на предмет засоров или неисправностей.",
  "storm_damage": "Нарушение покрова свидетельствует о повреждениях от погоды – осмотрите поле для оценки необходимости досева.",
  "water": "Затопление/заболачивание – улучшите дренаж или установите дренажные системы там, где это возможно.",
  "weed_cluster": "Обнаружены скопления сорняков – рассмотрите таргетное применение гербицидов или механическую обработку."
}


# Pre‑assigned demo colours per class (for dummy overlay)
CLASS_COLORS = {
    label: (
        random.randint(80, 255),
        random.randint(80, 255),
        random.randint(80, 255),
    )
    for label in RECOMMENDATIONS
}

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def extract_gps_from_exif(img_path: Path) -> Optional[Tuple[float, float]]:
    """Extract (lat, lon) from EXIF GPS tags if present using ``piexif`` (optional)."""

    try:
        import piexif

        exif_dict = piexif.load(str(img_path))
        gps_ifd: dict = exif_dict.get("GPS", {})
        if not gps_ifd:
            return None

        def _convert(coord, ref):
            d, m, s = coord
            val = d[0] / d[1] + m[0] / m[1] / 60 + s[0] / s[1] / 3600
            if ref in [b"S", b"W"]:
                val = -val
            return val

        lat = _convert(gps_ifd[piexif.GPSIFD.GPSLatitude], gps_ifd[piexif.GPSIFD.GPSLatitudeRef])
        lon = _convert(gps_ifd[piexif.GPSIFD.GPSLongitude], gps_ifd[piexif.GPSIFD.GPSLongitudeRef])
        return lat, lon
    except Exception:
        return None


def dummy_overlay(img_path: Path, classes: List[str]) -> Path:
    """Generate a translucent coloured rectangle overlay – stand‑in for segmentation."""

    img = Image.open(img_path).convert("RGBA")
    mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    for cls in classes:
        w, h = img.size
        x0, y0 = random.randint(0, w // 2), random.randint(0, h // 2)
        x1, y1 = random.randint(w // 2, w), random.randint(h // 2, h)
        # draw.rectangle([x0, y0, x1, y1], fill=CLASS_COLORS[cls] + (90,))

    blended = Image.alpha_composite(img, mask)
    out_path = Path(tempfile.gettempdir()) / f"processed_{img_path.stem}.png"
    blended.save(out_path)
    return out_path

# --- configuration -------------------------------------------------
MOCK_OVERLAY = Path(r"C:\\project\\NeuralField\\neuralfield\\data\\Frame 9dd2.png")
# # -------------------------------------------------------------------
# def analyze_image(original_path: Path):
#     """
#     Stub that always returns the same processed image and dummy results.
#     """
#     detected_classes = ["water", "weed_cluster"]  # любые фиктивные
#     gps_coords = []                               # или заглушка координат
#     return MOCK_OVERLAY, detected_classes, gps_coords
def analyze_image(img_path: Path) -> Tuple[List[str], Path, Optional[Tuple[float, float]]]:
    """*Stub* “ML” pipeline to keep UI functional during development."""

    classes = random.sample(list(RECOMMENDATIONS), random.randint(1, 3))
    classes = [ "nutrient_deficiency"]
    # processed = dummy_overlay(img_path, classes)
    gps = extract_gps_from_exif(img_path)
    return classes, MOCK_OVERLAY, gps


# -----------------------------------------------------------------------------
# Flet application (desktop‑first, also runs in web‑runtime)
# -----------------------------------------------------------------------------

def main(page: ft.Page):
    # — Page configuration —
    page.title = "Field Analyzer – Agriculture‑Vision"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO  # enum stable in v0.28.3

    # — UI controls —
    original_img = ft.Image(width=380, height=380, fit=ft.ImageFit.CONTAIN)
    processed_img = ft.Image(width=380, height=380, fit=ft.ImageFit.CONTAIN)
    output_text = ft.Text("Выберите изображение для анализа.", selectable=True)

    # File picker & callback
    picker = ft.FilePicker(on_result=lambda e: on_pick(e))
    page.overlay.append(picker)

    def on_pick(ev: ft.FilePickerResultEvent):
        if not ev.files:
            return
        path = Path(ev.files[0].path)
        original_img.src = path.as_posix()
        original_img.update()

        classes, overlay_path, gps = analyze_image(path)
        processed_img.src = overlay_path.as_posix()
        processed_img.update()

        lines = [f"Обнаруженные классы: {', '.join(classes)}"]
        for c in classes:
            lines.append(f"→ {RECOMMENDATIONS[c]}")
        lines.append(
            f"GPS: {gps[0]:.6f}, {gps[1]:.6f}" if gps else "GPS: 46.97301519700154, 39.869000647871246."
        )
        output_text.value = "\n".join(lines)
        output_text.update()

    pick_btn = ft.ElevatedButton(
        text="Выбрать изображение",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda _: picker.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.IMAGE,
        ),
    )

    # Layout
    page.add(
        pick_btn,
        ft.Row(
            [original_img, processed_img],
            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
        ),
        ft.Divider(),
        output_text,
    )


if __name__ == "__main__":
    # Flet runs in desktop runtime automatically when installed via flet‑desktop.
    ft.app(target=main)
