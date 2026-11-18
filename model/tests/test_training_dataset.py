"""두 클래스 라벨 및 학습 연결을 검증합니다."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from scripts.check_dataset import validate_dataset
from scripts import train


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.data = self.base / 'data.yaml'
        self.data.write_text(yaml.safe_dump({
            'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
            'nc': 2, 'names': {0: 'bicycle', 1: 'humans'}}), encoding='utf-8')
        for split in ('train', 'val', 'test'):
            (self.base / 'images' / split).mkdir(parents=True)
            (self.base / 'labels' / split).mkdir(parents=True)
            Image.new('RGB', (16, 16)).save(self.base / 'images' / split / 'sample.jpg')
            (self.base / 'labels' / split / 'sample.txt').write_text(
                '0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n', encoding='utf-8')

    def test_both_classes_and_empty_labels_are_valid(self):
        (self.base / 'labels/val/sample.txt').write_text('', encoding='utf-8')
        report = validate_dataset(self.data)
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['objects']['train'], {0: 1, 1: 1})
        self.assertEqual(report['empty_labels']['val'], 1)

    def test_unknown_class_is_rejected(self):
        (self.base / 'labels/train/sample.txt').write_text('2 0.5 0.5 0.2 0.2\n', encoding='utf-8')
        self.assertTrue(validate_dataset(self.data)['errors'])

    def test_missing_label_and_invalid_box_are_rejected(self):
        (self.base / 'labels/train/sample.txt').unlink()
        (self.base / 'labels/val/sample.txt').write_text('1 0.95 0.5 0.5 0.5\n', encoding='utf-8')
        self.assertEqual(len(validate_dataset(self.data)['errors']), 2)

    def test_check_only_does_not_load_model(self):
        with patch('sys.argv', ['train.py', '--data', str(self.data), '--check-only']), \
             patch.object(train, 'YOLO') as model:
            train.main()
        model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
