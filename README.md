# Система анализа данных с видеокамер на предприятии

## Описание

Система представляет собой проект на Python3 с использованием фреймворка YOLO для решения задач компьютерного зрения. Умеет определять объекты (конкретный пример - поезд) и положение человека (сидя, стоя, в движении, в "наклоне"). Детектор основан на предобученных моделях COCO.

С использованием фреймворка Streamlit создана среда мониторинга с обзором видеоряд в реальном времени и объекты, запечатленные на видео. Взаимодействие между системой обработки видео и дэшбордом реализовано при помощи TCP-сокетов.

Оптимизационные решения:
- изменение размера изображения (кадра потока) и анализ каждого N кадра;
- динамическая система определения зон, где маловероятно присутствие человека.

## Установка

```
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo mv rus.traineddata /usr/share/tesseract-ocr/5/tessdata/ 
pip install -r requirements.txt
```

## Зависимости

```
ultralytics>=8.2.0
opencv-python>=4.10.0
psycopg2-binary>=2.9.9
streamlit>=1.32.0
pandas>=2.2.0
plotly>=5.18.0
numpy>=1.26.0
supervision>=0.23.0
torch>=2.0.0
torchvision>=0.15.0
pytesseract>=0.3.13
```

## Запуск

```
export PATH="$PATH:/home/dev/.local/bin"
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
docker compose up -d
docker ps
streamlit run dashboard.py
python3 run_system.py
```

После запуска детектора и дэшборда в среде мониторинга необходимо инициировать соединение с системой обработки видеоряда (панель управления слева).