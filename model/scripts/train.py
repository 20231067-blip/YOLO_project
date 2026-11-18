"""데이터 YAML의 클래스 정의에 맞춰 YOLO11n을 추가 학습합니다."""
import argparse
from pathlib import Path

from ultralytics import YOLO
import yaml

if __package__:
    from .check_dataset import DEFAULT_DATA, require_valid_dataset
else:
    from check_dataset import DEFAULT_DATA, require_valid_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DEFAULT_DATA))
    p.add_argument('--weights', default=str(ROOT / 'yolo11n.pt'))
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu', help='CPU는 cpu, NVIDIA GPU는 0')
    p.add_argument('--name', default='weather_person_v1')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--plots', action='store_true', help='학습 그래프와 샘플 이미지를 저장합니다.')
    p.add_argument('--check-only', action='store_true', help='데이터만 검사하고 학습하지 않습니다.')
    a = p.parse_args()
    if a.epochs < 1 or a.batch < 1 or a.imgsz < 1 or a.workers < 0:
        p.error('epochs, batch, imgsz는 양수이고 workers는 0 이상이어야 합니다.')
    try:
        report = require_valid_dataset(a.data)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        p.exit(1, f'{exc}\n')
    if a.check_only:
        return
    weights = Path(a.weights).expanduser()
    if not weights.is_absolute() and not weights.is_file() and (ROOT / weights).is_file():
        weights = ROOT / weights
    model = YOLO(str(weights))
    model.train(data=report['data_file'], epochs=a.epochs, batch=a.batch, imgsz=a.imgsz,
                device=a.device, workers=a.workers, project=str(ROOT / 'runs'),
                name=a.name, seed=42, patience=12, plots=a.plots)
    print('학습 결과 위치:', model.trainer.save_dir)
    print('사용할 모델:', Path(model.trainer.save_dir) / 'weights/best.pt')


if __name__ == '__main__':
    main()
