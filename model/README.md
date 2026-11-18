# RightTurn Weather + Camera (우회전 보행자 감지 연구·교육용 시제품)

## 1. 목표
YOLO11n을 추가 학습하고 웹캠, USB 카메라, 영상 파일 또는 RTSP/IP 카메라에 적용하는 시연 프로젝트입니다. 현재 저장소의 Roboflow 이미지 61장과 라벨 61개를 model/data/pedestrian으로 가져왔습니다. 주간/야간/우천별 현장 데이터는 별도로 추가해야 합니다.

**중요:** 보행자 신호는 **시연자가 수동으로 입력**합니다(`g`: 녹색, `r`: 적색, `u`: 알 수 없음). 적색 신호 진입은 법적 판단이 아닌 *의심 상황*입니다. 이 프로젝트는 **인증된 교통 안전 장치가 아니며**, 화면 표시를 우회전해도 된다는 허가로 해석해서는 안 됩니다.

## 2. 설치

```powershell
cd RightTurn-Weather-Camera
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

카메라 화면의 한글은 Pillow로 표시합니다. Windows에서는 기본 제공되는 맑은 고딕을 사용합니다. 다른 운영체제에서는 나눔고딕, Noto Sans CJK 또는 AppleGothic 글꼴이 필요합니다.

## 3. 학습 데이터

명령은 이 `model` 폴더에서 필요한 패키지가 설치된 Python 환경으로 실행합니다.
원본 `pedestrian crosswalk detection.v2i.yolov11` 폴더는 보존하고 아래 구조로 복사했습니다.
원본의 `valid` 분할은 `val`로 맞췄습니다.

```text
data/pedestrian/
  data.yaml
  images/train/   # 43장
  images/val/     # 12장
  images/test/    # 6장
  labels/train/  # 43개
  labels/val/    # 12개
  labels/test/   # 6개
```

클래스는 **0: bicycle, 1: humans**입니다. 사람을 0으로 바꾸면 자전거와 섞이므로 원본 번호를 유지합니다.
YOLO 라벨 한 줄은 `클래스번호 중심_x 중심_y 너비 높이`이며 좌표는 0`1로 정규화합니다.
객체가 없는 이미지는 같은 이름의 빈 `.txt` 파일을 둡니다.
카메라 시연은 학습한 `humans`를 보행자로 추적하며 기본 COCO 모델의 `person`도 지원합니다.

라이선스와 출처는 `data/pedestrian/SOURCE.md`, `README.dataset.txt`, `README.roboflow.txt`에 보존했습니다.
학습 이미지와 라벨은 Git 제외 규칙을 제거해 소스트리에서 관리할 수 있습니다.
원본 영상, `runs/`, 모델 `*.pt`, 라벨 캐시 `*.cache`는 계속 제외합니다.

이 데이터는 61장의 소규모 샘플이며 파일명상 같은 영상의 프레임이 여러 분할에 포함됩니다.
최종 성능은 독립된 촬영 영상 및 주간/야간/우천별 데이터로 평가해야 합니다.
직접 데이터를 추가할 때는 촬영 영상 단위로 train/val/test를 먼저 분리하고 같은 클래스 정의로 라벨링하세요.

```powershell
python scripts/extract_frames.py --video data/raw/night/night01.mp4 --out data/staging/night01 --every-sec 2
```

## 4. 데이터 검사 및 추가 학습

```powershell
# 이미지 읽기, 라벨 대응, 클래스 및 좌표 검사
python scripts/check_dataset.py

# 학습하지 않고 학습 코드의 데이터 연결만 확인
python scripts/train.py --check-only

# CPU로 짧은 실행 확인
python scripts/train.py --epochs 1 --batch 2 --imgsz 320 --device cpu --name dataset_smoke

# NVIDIA GPU 환경이 준비된 경우 본 학습
python scripts/train.py --epochs 40 --batch 8 --device 0 --name weather_person_v1
```

GPU가 없으면 본 학습에도 `--device cpu --batch 2`를 사용할 수 있지만 오래 걸릴 수 있습니다.
기본 데이터는 `data/pedestrian/data.yaml`이고 다른 YAML은 `--data 경로/data.yaml`로 지정합니다.
검사기는 `images/<분할>`, `labels/<분할>` 구조를 사용합니다.
YAML의 `path`는 생략하거나 절대 경로로 지정하세요. 기본 설정에는 절대 경로가 없어 프로젝트를 옮겨도 동작합니다.
기존 모델에서 더 학습하려면 `--weights 경로/best.pt`, 학습 그래프를 저장하려면 `--plots`를 추가합니다.

완료 시 실제 결과 폴더가 출력됩니다. 같은 이름이 있으면 `weather_person_v12`처럼 새 이름이 만들어질 수 있습니다.
아래는 본 학습 결과가 `runs/weather_person_v1`에 생성된 경우입니다.

```powershell
python scripts/evaluate.py --weights runs/weather_person_v1/weights/best.pt
python run_demo.py --source data/raw/crosswalk.mp4 --model runs/weather_person_v1/weights/best.pt
python run_demo.py --source 0 --model runs/weather_person_v1/weights/best.pt
```

`dataset_smoke`는 학습 연결을 확인하기 위한 모델이며 성능 검증을 마친 최종 모델이 아닙니다.

## 5. 카메라 실행

현재 제공된 영상은 data/raw/crosswalk.mp4에 있습니다. Windows에서는 프로젝트 폴더에서 다음 명령을 실행하면 SSL과 필수 패키지가 정상인 파이썬 환경을 찾아 실행합니다.

```powershell
python run_demo.py
```

VS Code에서는 실행 및 디버그에서 **횡단보도 영상 실행**을 선택할 수 있습니다. 기본 모델은 프로젝트 폴더의 yolo11n.pt를 읽습니다. 모델을 처음 내려받을 때 SSL 모듈 관련 오류가 발생하면 파이썬 환경 문제이므로 SSL이 정상인 환경으로 실행하세요.


영상의 상대 경로는 현재 실행 폴더, 프로젝트 폴더, data/raw 폴더, scripts 폴더 순서로 찾습니다. 파일이 없으면 확인한 경로를 오류 메시지로 보여 줍니다. `data/raw/crosswalk.mp4`는 프로젝트 폴더에서 다음 명령으로 실행할 수 있습니다.

```powershell
python scripts/demo_camera.py --source data/raw/crosswalk.mp4
```


```powershell
# 노트북 내장 카메라
python scripts/demo_camera.py --source 0 --model runs/weather_person_v1/weights/best.pt
# USB 카메라는 장치 번호가 1일 수 있습니다.
python scripts/demo_camera.py --source 1 --model runs/weather_person_v1/weights/best.pt --roi-file config/usb_roi.json
# 동영상 파일
python scripts/demo_camera.py --source data/raw/day/day01.mp4 --model runs/weather_person_v1/weights/best.pt --roi-file config/video_roi.json
# RTSP 카메라(접근 권한이 있는 실제 주소를 사용하세요).
python scripts/demo_camera.py --source "rtsp://USERNAME:PASSWORD@HOST:554/stream" --model runs/weather_person_v1/weights/best.pt --roi-file config/ip_roi.json
```

시작할 때 횡단보도를 사각형으로 지정하고 Enter 키로 확정합니다. 관심 영역은 --roi-file로 지정한 파일에 정규화된 좌표로 저장됩니다. **z** 키로 다시 지정할 수 있습니다. 카메라나 촬영 각도가 다르면 **서로 다른 관심 영역 파일을 사용**합니다. 미리보기는 크기를 조절하지만 감지는 원본 프레임에서 수행합니다. --output recorded.mp4를 지정하면 표시 화면을 원래 해상도로 저장할 수 있습니다. **RTSP 접속 정보나 촬영 영상을 GitHub에 올리지 마세요.**


### 노트북 내장 카메라 연결

Windows 카메라 앱과 화상회의 앱을 종료한 뒤 프로젝트 폴더에서 실행합니다.

```powershell
python run_demo.py --source 0 --roi-file config/webcam_roi.json
```

Windows에서는 DirectShow, Media Foundation, 기본 연결 방식을 순서대로 시도합니다. 연결 확인만 하려면 다음 명령을 사용합니다.

```powershell
python run_demo.py --source 0 --check-camera
```

특정 연결 방식은 --camera-backend dshow 또는 --camera-backend msmf로 지정합니다. 일반 카메라와 적외선 카메라의 장치 번호가 다르면 --source 1도 확인합니다. 카메라 앱은 정상인데 파이썬만 실패할 경우 Windows의 카메라 설정에서 데스크톱 앱의 카메라 접근 허용 여부를 확인합니다.

## 6. 권장 시험 사례

- 주간: 보행자 없음 / 대기 중 / 횡단 중
- 야간: 어두운 환경의 보행자 / 역광 또는 전조등 눈부심
- 우천: 우산 / 반사 / 젖은 노면 / 일부 가림
- 신호: 시연자가 g/r/u 키로 수동 설정하고 상태 처리 로직**만** 확인
- 카메라: 웹캠과 저장 영상을 각각 시험하고 지연, 연결 끊김, 프레임 누락 확인

객체 감지 지표는 정밀도(P), 재현율(R), 평균 정밀도(mAP50/mAP50-95)입니다. 시스템 지표는 횡단 감지 재현율, 오경고 빈도, 경고 지연 시간입니다. 카메라 연결이 끊기거나 결과가 불확실할 때 화면에 우회전이 안전하다고 표시해서는 안 됩니다.

## 7. 한계 및 향후 과제

현재 보행자 신호는 영상에서 자동으로 판별하지 않습니다. 별도의 신호 인식 기능에는 전용 지도 학습 데이터셋과 평가가 필요합니다. 사각형 관심 영역과 발 위치를 이용한 추정은 특이한 촬영 각도에서 부정확할 수 있습니다. 이 프로젝트는 실제 도로에서 운영하는 안전 시스템이 아닌 *실험실 시연용*입니다.

공식 참고 자료:
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/modes/track/
- https://docs.ultralytics.com/datasets/detect/
