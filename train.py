from pathlib import Path

import torch
from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "datasets" / "Dataset fruit.v1i.yolov8" / "data.yaml"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"Пристрій навчання: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Пристрій навчання: CPU (CUDA недоступна)")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Не знайдено data.yaml: {DATA_YAML}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=2,
        workers=2,
        device=device,
        name="yolo_auto",
    )

    print(f"Навчання завершено. Результати: {Path('runs/detect/yolo_auto')}")


if __name__ == "__main__":
    main()
