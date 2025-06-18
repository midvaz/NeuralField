
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import time
import random
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import flet as ft
from PIL import Image

from neuralfield.interface.test import neurnolAnalize
from neuralfield.interface.const import LANGUAGES, RECOMMENDATIONS, DATA, PLACEHOLDER_IMG


def analyze_image(img_path: Path) -> Tuple[List[str], Path, Optional[Tuple[float, float]]]:
    key = img_path.as_posix()
    if key in DATA:
        resp = DATA[key]
        return resp["classes"], resp["overlay"], resp.get("gps")
    classes = [random.choice(list(RECOMMENDATIONS))]
    overlay = Path(tempfile.gettempdir()) / f"processed_{img_path.stem}.png"
    Image.open(img_path).convert("RGBA").save(overlay)
    return classes, neurnolAnalize(overlay), None

# -----------------------------------------------------------------------------
# Application class
# -----------------------------------------------------------------------------
class FieldAnalyzerApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.lang = "ru"
        self.dark = False
        self._apply_locale()
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self._setup_ui()

    def _apply_locale(self):
        texts = LANGUAGES[self.lang]
        self.page.title = texts["app_title"]

    def _setup_ui(self):
        texts = LANGUAGES[self.lang]
        # Theme & Language controls
        self.lang_dropdown = ft.Dropdown(
            label=texts["language_label"],
            value=self.lang,
            options=[ft.dropdown.Option(k) for k in LANGUAGES],
            on_change=self._on_language_change
        )
        self.theme_switch = ft.Switch(
            label=texts["theme_label"],
            value=self.dark,
            on_change=self._on_theme_change
        )
        # File picker
        self.picker = ft.FilePicker(on_result=self.on_pick)
        self.page.overlay.append(self.picker)
        # Buttons
        self.pick_btn = ft.ElevatedButton(
            text=texts["btn_select"],
            icon=ft.Icons.FOLDER_OPEN,
            on_click=lambda _: self.picker.pick_files(
                allow_multiple=False,
                file_type=ft.FilePickerFileType.IMAGE
            ),
        )
        self.export_btn = ft.ElevatedButton(
            text=texts["btn_export"],
            icon=ft.Icons.DOWNLOAD,
            on_click=self._on_export
        )
        self.config_btn = ft.ElevatedButton(
            text=texts["btn_config"],
            icon=ft.Icons.EDIT,
            on_click=self._on_config
        )
        # Images and output
        self.original_img = ft.Image(src=PLACEHOLDER_IMG.as_posix(), width=380, height=380, fit=ft.ImageFit.CONTAIN)
        self.processed_img = ft.Image(src=PLACEHOLDER_IMG.as_posix(), width=380, height=380, fit=ft.ImageFit.CONTAIN)
        self.output_text = ft.Text(texts["mm_select"], selectable=True)
        # Layout
        self.page.controls.clear()
        self.page.add(
            ft.Row([self.lang_dropdown, self.theme_switch]),
            ft.Row([self.pick_btn, self.export_btn, self.config_btn]),
            ft.Row([self.original_img, self.processed_img], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
            ft.Divider(),
            self.output_text
        )
        self.page.update()

    def _on_language_change(self, e: ft.DropdownChangeEvent):
        self.lang = e.control.value
        self._apply_locale()
        self._setup_ui()

    def _on_theme_change(self, e: ft.SwitchEvent):
        self.dark = e.control.value
        self.page.theme_mode = ft.ThemeMode.DARK if self.dark else ft.ThemeMode.LIGHT
        self.page.update()

    def on_pick(self, ev: ft.FilePickerResultEvent):
        if not ev.files:
            return
        img_path = Path(ev.files[0].path)
        self.original_img.src = img_path.as_posix()
        self.original_img.update()
        time.sleep(1)
        classes, overlay, gps = analyze_image(img_path)
        self.processed_img.src = overlay.as_posix()
        self.processed_img.update()
        self._update_text(classes, gps)

    def _update_text(self, classes: List[str], gps: Optional[Tuple[float, float]]):
        texts = LANGUAGES[self.lang]
        lines = [f"{texts['lbl_classes']}{', '.join(classes)}"]
        for c in classes:
            lines.append(f"→ {RECOMMENDATIONS.get(c, '')}")
        lines.append(f"GPS: {gps[0]:.6f}, {gps[1]:.6f}" if gps else texts['gps_missing'])
        self.output_text.value = "".join(lines)
        self.output_text.update()

    def _on_export(self, e: ft.ControlEvent):
        # Initiate FilePicker to select save location
        self.save_picker = ft.FilePicker(on_result=self._save_report)
        self.page.overlay.append(self.save_picker)
        self.page.update()
        self.save_picker.save_file(
    dialog_title=LANGUAGES[self.lang]['btn_export'],
    file_name="report.json",
    allowed_extensions=["json"]
)
    def _save_report(self, ev: ft.FilePickerResultEvent):
        if not getattr(ev, "path", None):
            return
        save_path = Path(ev.path)

        lines = self.output_text.value.splitlines()
        report = {
            'classes': lines[0]
                .replace(LANGUAGES[self.lang]['lbl_classes'], '')
                .strip()
                .split(', '),
            'recommendations': lines[1:],
            'gps': None if LANGUAGES[self.lang]['gps_missing'] in self.output_text.value
                   else self._last_gps 
        }
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            dlg = ft.AlertDialog(
                title=ft.Text(LANGUAGES[self.lang]['msg_export_success_title']),
                content=ft.Text(LANGUAGES[self.lang]['msg_export_success']),
                actions=[ft.TextButton("OK", on_click=lambda e: self._close_dialog())]
            )
            self._show_dialog(dlg)

        except Exception as err:
            dlg = ft.AlertDialog(
                title=ft.Text("Error"),
                content=ft.Text(f"Не удалось сохранить отчёт: {err}"),
                actions=[ft.TextButton("OK", on_click=lambda e: self._close_dialog())]
            )
            self._show_dialog(dlg)

    def _show_dialog(self, dlg: ft.AlertDialog):
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()

    def _close_dialog(self, e: ft.ControlEvent = None):
        if self.page.dialog:
            self.page.dialog.open = False
            self.page.update()

    def _on_config(self, e: ft.ControlEvent):
        txt = ft.TextField(
            value=json.dumps(RECOMMENDATIONS, ensure_ascii=False, indent=2),
            multiline=True,
            width=600,
            height=400
        )
        dlg = ft.AlertDialog(
            title=ft.Text(LANGUAGES[self.lang]['btn_config']),
            content=txt,
            actions=[
                ft.TextButton("Save", on_click=lambda _: self._save_config(txt, dlg)),
                ft.TextButton("Cancel", on_click=lambda _: dlg.close())
            ]
        )
        self.page.dialog = dlg
        dlg.open = True

    def _save_config(self, txt: ft.TextField, dlg: ft.AlertDialog):
        try:
            data = json.loads(txt.value)
            RECOMMENDATIONS.clear()
            RECOMMENDATIONS.update(data)
        except json.JSONDecodeError:
            pass
        dlg.open = False


def main(page: ft.Page):
    FieldAnalyzerApp(page)

if __name__ == "__main__":
    ft.app(target=main)
