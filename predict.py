import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("runs/detect/yolo_auto-2/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Перевірка навченої моделі YOLOv8n на своєму зображенні")
    parser.add_argument("--source", required=True, help="Шлях до зображення або папки")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Шлях до вагів моделі")
    parser.add_argument("--conf", type=float, default=0.25, help="Поріг впевненості")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Ваги не знайдено: {weights}\nСпочатку запустіть навчання (python train.py)."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Пристрій: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(str(weights))
    results = model.predict(
        source=args.source,
        device=device,
        conf=args.conf,
        save=True,
        project="runs/detect",
        name="predict",
        exist_ok=True,
    )

    for r in results:
        print(f"{r.path}: {len(r.boxes)} об'єктів")
    print("Результати збережено в runs/detect/predict")


if __name__ == "__main__":
    main()
