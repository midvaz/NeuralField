from typing import Dict, Any
from pathlib import Path

from PIL import Image
# -----------------------------------------------------------------------------
# Localization and Themes
# -----------------------------------------------------------------------------
LANGUAGES = {
    "ru": {
        "app_title": "Field Analyzer – Agriculture‑Vision",
        "btn_select": "Выбрать изображение",
        "mm_select": "Выберите изображение для анализа.",
        "btn_export": "Сохранить отчёт",
        "btn_config": "Настроить советы",
        "lbl_classes": "Обнаруженные классы: ",
        "gps_missing": "GPS: отсутствует",
        "language_label": "Язык",
        "theme_label": "Тема",
    },
    "en": {
        "app_title": "Field Analyzer – Agriculture‑Vision",
        "btn_select": "Select Image",
        "mm_select": "Select an image to analyze.",
        "btn_export": "Save Report",
        "btn_config": "Configure Recommendations",
        "lbl_classes": "Detected classes: ",
        "gps_missing": "GPS: unavailable",
        "language_label": "Language",
        "theme_label": "Theme",
    }
}

# -----------------------------------------------------------------------------
# Agriculture‑Vision label → recommendation mapping
# -----------------------------------------------------------------------------
RECOMMENDATIONS: Dict[str, str] = {
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

# -----------------------------------------------------------------------------
# Mock responses storage: input image path → fixed output
# -----------------------------------------------------------------------------
DATA: Dict[str, Dict[str, Any]] = {
    Path(r"C:/project/NeuralField/neuralfield/data/3CPYLP42D_2562-1149-3074-1661.jpg").as_posix(): {
        "classes": ["bare_soil", "weed_cluster"],
        "overlay": Path(r"C:/project/NeuralField/tests/data/preprocces/Frame92.png"),
        "gps": (46.973015, 39.869001)
    },
    Path(r"C:/project/NeuralField/neuralfield/data/2Z8X92VXV_621-2701-1133-3213.jpg").as_posix(): {
        "classes": ["water", "storm_damage"],
        "overlay": Path(r"C:/project/NeuralField/tests/data/preprocces/Frame95.png"),
        "gps": None
    },
    
    Path(r"C:\project\NeuralField\neuralfield\data\1LJQVM8TW_766-7277-1278-7789.jpg").as_posix(): {
        "classes": ["water", "storm_damage"],
        "overlay": Path(r"C:\project\NeuralField\tests\data\preprocces\Frame 91.png"),
        "gps": None
    },
    Path(r"C:\project\NeuralField\neuralfield\data\3CPYLP42D_4598-3940-5110-4452.jpg").as_posix(): {
        "classes": ["planter_skip"],
        "overlay": Path(r"C:\project\NeuralField\tests\data\preprocces\Frame 93.png"),
        "gps": None
    },
    Path(r"C:\project\NeuralField\neuralfield\data\3VVJATQ4D_2918-606-3430-1118.jpg").as_posix(): {
        "classes": ["planter_skip", "nutrient_deficiency"],
        "overlay": Path(r"C:\project\NeuralField\tests\data\preprocces\Frame 94.png"),
        "gps": None
    },
    Path(r"C:\project\NeuralField\neuralfield\data\D83BCIF7K_8243-3715-8755-4227.jpg").as_posix(): {
        "classes": ["planter_skip", "nutrient_deficiency"],
        "overlay": Path(r"C:\project\NeuralField\tests\data\preprocces\3.png"),
        "gps": None
    },
}

# -----------------------------------------------------------------------------
# Default placeholder before selection
# -----------------------------------------------------------------------------
PLACEHOLDER_IMG: Path = Path(r"C:\project\NeuralField\neuralfield\mock_2.png")
if not PLACEHOLDER_IMG.exists():
    img = Image.new("RGB", (380, 380), color=(230, 230, 230))
    img.save(PLACEHOLDER_IMG)
