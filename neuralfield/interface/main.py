
from pathlib import Path

import flet as ft


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 650

# TODO: НУЖНО СЧИТАТЬ ЭТИ ЗНАЧЕНИЯ ДИНАМИЧЕСКИ, В ЗАВИСИМОСТИ ОТ РАЗМЕРА ОКНА
IMG_WIDTH = 500
IMG_HEIGHT = 500

base_dir = Path(__file__).parent  # Текущая директория скрипта
file_name = "mock.png"

MOCK_SELECT_IMG_PATH = str(base_dir / ".." / "data" / file_name)

MESSAGE_MOCK_RESULT = "Тут будет результат обработки"
MESSAGE_SELECT_PATH = "Выбор изображения"


def main_windows(page: ft.Page):
    page.title = "NeuralField"

    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.theme = ft.theme.Theme(color_scheme_seed="black")
    page.window.width = WINDOW_WIDTH
    page.window.height = WINDOW_HEIGHT
    
    def on_upload(e: ft.FilePickerResultEvent):
        if e.files:
            # Открываем изображение из локального пути
            #TODO: вызываем обрабоку изображения
            uploaded_image.src = e.files[0].path
            uploaded_image.width = IMG_WIDTH 
            uploaded_image.height = IMG_HEIGHT  
            uploaded_image.update()
            
            #TODO: вызываем получение текста
            text_right.value = "Какой-то результат обработки"
            text_right.update()

    # Создаем кнопку загрузки и подключаем FilePicker
    file_picker = ft.FilePicker(on_result=on_upload)
    page.overlay.append(file_picker)

    # Добавляем кнопку загрузки
    upload_button = ft.ElevatedButton(
        "Загрузить изображение", 
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False
        )
    )

    # Создаем объект для отображения изображения
    uploaded_image = ft.Image(
        src=MOCK_SELECT_IMG_PATH,
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
    )

    # Добавляем элементы на страницу
    left_column = ft.Column(
        controls=[
            upload_button,
            uploaded_image
        ],
        alignment="center",
        spacing=20 
    )
    
    # Объект для отображение текста 
    text_right = ft.Text(
        value=MESSAGE_MOCK_RESULT,
        size=48,
        width=400,
        height=400,
        color=ft.colors.WHITE,
    )
    
    row_layout = ft.Row(
        controls=[
            left_column, 
            text_right      
        ],
        alignment="center",  
        spacing=20           
    )

    page.add(row_layout)
    page.update()

    
if __name__ == '__main__':
    ft.app(target=main_windows) 
    