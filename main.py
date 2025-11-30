from video_processor import VideoProcessor
import argparse


def main():
    parser = argparse.ArgumentParser(description='Factory Video Analysis')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')

    args = parser.parse_args()

    print("🚀 Запуск системы анализа видео...")

    # Инициализация и обработка видео
    processor = VideoProcessor(args.video)
    processor.process_video()

    print("✅ Обработка завершена!")


if __name__ == "__main__":
    main()