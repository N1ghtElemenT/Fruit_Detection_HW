import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")

WEIGHTS = Path("runs/detect/yolo_auto-2/weights/best.pt")
DATA_YAML = Path("datasets/Dataset fruit.v1i.yolov8/data.yaml")


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Ваги не знайдено: {WEIGHTS}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml не знайдено: {DATA_YAML}")

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"Пристрій: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Пристрій: CPU (CUDA недоступна)")

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        workers=0,
        device=device,
        name="yolo_auto_test",
        exist_ok=True,
        save_json=False,
    )

    print("\n=== Метрики на тестовій вибірці ===")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print(f"mAP50:        {metrics.box.map50:.4f}")
    print(f"mAP50-95:     {metrics.box.map:.4f}")

    names = metrics.names
    print("\n=== По кожному класу ===")
    print(f"{'Клас':<15} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    for i, name in names.items():
        print(
            f"{name:<15} {metrics.box.p[i]:>10.4f} {metrics.box.r[i]:>10.4f} "
            f"{metrics.box.ap50[i]:>10.4f} {metrics.box.ap[i]:>10.4f}"
        )

    print(f"\nРезультати та графіки збережено в: {metrics.save_dir}")


if __name__ == "__main__":
    main()
