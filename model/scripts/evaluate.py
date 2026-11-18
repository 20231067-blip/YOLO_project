"""학습에 사용하지 않은 test 분할로 학습 모델을 평가합니다."""
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
    p.add_argument('--weights', required=True)
    p.add_argument('--data', default=str(DEFAULT_DATA))
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    try:
        report = require_valid_dataset(a.data, required_splits=('test',))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        p.exit(1, f'{exc}\n')
    weights = Path(a.weights).expanduser()
    if not weights.is_absolute() and not weights.is_file() and (ROOT / weights).is_file():
        weights = ROOT / weights
    metrics = YOLO(str(weights)).val(data=report['data_file'], split='test', device=a.device,
                                    project=str(ROOT / 'runs'), name='test_evaluation', plots=False)
    print(f'정밀도={metrics.box.mp:.4f} 재현율={metrics.box.mr:.4f} '
          f'mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
