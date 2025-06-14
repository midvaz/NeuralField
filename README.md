# NeuralField
Приложение для сегментирования изображений полей при помощи неройнной сети Unet/vit

# Установка 

## Установка проектного менеджера 
```bash
in Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

in Windows 
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv init
```

## Установка пакетов
```bash
uv add requests

uv add -r requirements.txt
```

## Запуск
```bash
uv run ./neuralfield/interface/main.py
```



pip3 install -r neuralfield/network/vit/reqirements.txt