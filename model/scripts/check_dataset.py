"""YAML 클래스, 이미지와 라벨의 대응 및 YOLO 경계 상자를 검사합니다."""
import argparse
from collections import Counter
import math
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/pedestrian/data.yaml'
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def validate_dataset(data_file=DEFAULT_DATA, required_splits=('train', 'val')):
    data_file = Path(data_file).expanduser().resolve()
    config = yaml.safe_load(data_file.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('데이터 YAML은 매핑 형식이어야 합니다.')
    names = config.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(key): value for key, value in names.items()}
    else:
        raise ValueError('YAML에 names 클래스 정의가 필요합니다.')
    if not names or set(names) != set(range(len(names))):
        raise ValueError('클래스 번호는 0부터 연속되어야 합니다.')
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError('클래스 이름은 비어 있지 않은 문자열이어야 합니다.')
    if 'nc' in config and config['nc'] != len(names):
        raise ValueError('nc와 names의 클래스 수가 다릅니다.')
    base = Path(config.get('path') or data_file.parent).expanduser()
    if not base.is_absolute():
        raise ValueError('YAML의 path는 생략하거나 절대 경로를 사용하세요.')
    counts, objects, empty_labels, errors = {}, {}, {}, []
    for split in ('train', 'val', 'test'):
        value = config.get(split)
        if not isinstance(value, str) or not value:
            if split in required_splits:
                errors.append(f'{split}: YAML에 이미지 폴더가 필요합니다.')
            counts[split], objects[split], empty_labels[split] = 0, {}, 0
            continue
        image_dir = (base / value).resolve()
        if image_dir.parent.name != 'images':
            raise ValueError(f'{split}: images/<분할> 구조를 사용하세요: {image_dir}')
        label_dir = image_dir.parent.parent / 'labels' / image_dir.name
        images = sorted(p for p in image_dir.glob('*') if p.is_file() and p.suffix.lower() in EXTS)
        labels = sorted(label_dir.glob('*.txt'))
        counts[split] = len(images)
        if split in required_splits and not images:
            errors.append(f'{split}: 이미지가 없습니다: {image_dir}')
        stems = {p.stem for p in images}
        if len(stems) != len(images):
            errors.append(f'{split}: 확장자만 다르고 이름이 같은 이미지가 있습니다.')
        for label in labels:
            if label.stem not in stems:
                errors.append(f'{label}: 대응하는 이미지가 없습니다.')
        class_counts, empty = Counter(), 0
        for image in images:
            try:
                with Image.open(image) as loaded:
                    loaded.verify()
            except (OSError, ValueError) as exc:
                errors.append(f'{image}: 이미지 읽기 실패: {exc}')
            label = label_dir / f'{image.stem}.txt'
            if not label.exists():
                errors.append(f'{image}: 라벨이 없습니다. 객체가 없으면 빈 .txt가 필요합니다.')
                continue
            try:
                lines = label.read_text(encoding='utf-8-sig').splitlines()
            except (OSError, UnicodeError) as exc:
                errors.append(f'{label}: 라벨 읽기 실패: {exc}')
                continue
            empty += not lines
            for number, line in enumerate(lines, 1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError('클래스 번호와 좌표 네 개가 필요합니다.')
                    class_id = int(parts[0])
                    if str(class_id) != parts[0] or class_id not in names:
                        raise ValueError(f'정의되지 않은 클래스 번호: {parts[0]}')
                    x, y, w, h = map(float, parts[1:])
                    if not (all(math.isfinite(v) for v in (x,y,w,h))
                            and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                            and x-w/2 >= -1e-5 and x+w/2 <= 1+1e-5
                            and y-h/2 >= -1e-5 and y+h/2 <= 1+1e-5):
                        raise ValueError('경계 상자가 정규화된 이미지 범위를 벗어납니다.')
                    class_counts[class_id] += 1
                except ValueError as exc:
                    errors.append(f'{label}:{number}: {exc}')
        objects[split], empty_labels[split] = dict(class_counts), empty
    return {'data_file': str(data_file), 'names': names, 'counts': counts,
            'objects': objects, 'empty_labels': empty_labels, 'errors': errors}


def require_valid_dataset(data_file=DEFAULT_DATA, required_splits=('train', 'val')):
    report = validate_dataset(data_file, required_splits)
    print('데이터 YAML:', report['data_file'])
    print('클래스:', report['names'])
    print('이미지 수:', report['counts'])
    print('클래스별 객체 수:', report['objects'])
    if report['errors']:
        raise ValueError('\n'.join(report['errors'][:50]) + f"\n검사 실패: 오류 {len(report['errors'])}개")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=None, help='데이터 YAML 경로')
    p.add_argument('--data-dir', default=str(DEFAULT_DATA.parent), help='data.yaml이 있는 폴더')
    args = p.parse_args()
    try:
        require_valid_dataset(args.data or Path(args.data_dir) / 'data.yaml')
    except (OSError, ValueError, yaml.YAMLError) as exc:
        p.exit(1, f'{exc}\n')
    print('이미지 읽기 및 라벨 형식 검사 완료. 라벨의 의미는 별도로 확인하세요.')


if __name__ == '__main__':
    main()
