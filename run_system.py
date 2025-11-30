import subprocess
import time
import os


def run_system():
    print("🏭 Анализ видеозаписи на предприятии")

    print("1. Проверка PostgreSQL...")
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        time.sleep(3)
    except:
        print("База данных уже запущена")

    print("2. Обработка видео...")
    video_file = "train_stands.mp4"

    if os.path.exists(video_file):
        subprocess.run(["python3", "main.py", "--video", video_file])
    else:
        print(f"⚠️ Видеофайл {video_file} не найден!")
        return


if __name__ == "__main__":
    run_system()