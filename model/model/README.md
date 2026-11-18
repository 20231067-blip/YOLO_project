# RightTurn Weather + Camera (우회전 보행자 감지 연구·교육용 시제품)

## 1. 목표
팀이 직접 촬영한 **주간 / 야간 / 우천 사진**으로 **YOLO11n 보행자 감지 모델을 추가 학습**한 뒤, **웹캠, USB 카메라, 동영상 또는 RTSP/IP 카메라**에 적용합니다. 별도의 관심 영역(ROI)과 객체 추적기로 경고 화면을 제어합니다. 직접 라벨을 작성한 이미지를 추가하기 전에는 모델의 추가 학습을 진행하지 않습니다.

**중요:** 보행자 신호는 **시연자가 수동으로 입력**합니다(`g`: 녹색, `r`: 적색, `u`: 알 수 없음). 적색 신호 진입은 법적 판단이 아닌 *의심 상황*입니다. 이 프로젝트는 **인증된 교통 안전 장치가 아니며**, 화면 표시를 우회전해도 된다는 허가로 해석해서는 안 됩니다.

## 2. 설치

```powershell
cd RightTurn-Weather-Camera
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

카메라 화면의 한글은 Pillow로 표시합니다. Windows에서는 기본 제공되는 맑은 고딕을 사용합니다. 다른 운영체제에서는 나눔고딕, Noto Sans CJK 또는 AppleGothic 글꼴이 필요합니다.

## 3. 직접 데이터 수집 및 라벨 작성

원본 촬영 영상을 `data/raw/day`, `data/raw/night`, data/raw/rain에 저장합니다. **최종 카메라와 같은 고정 시점**에서 다양한 조명, 비, 반사, 우산 및 가림 상황을 촬영합니다. 촬영 동의를 받고, 식별 가능한 얼굴과 차량 번호판은 적절히 처리합니다. **학습 자료를 수집하기 위해 차도에 서거나 도로를 막지 마세요.**

**프레임을 추출하기 전에 촬영 영상을 분할합니다.** 예를 들어 서로 다른 원본 촬영 회차나 장소를 학습용(`train`), 검증용(`val`), 시험용(`test`)으로 나눕니다. 같은 원본 영상의 인접 프레임이 서로 다른 데이터 묶음에 들어가지 않도록 합니다. 촬영 조건(주간/야간/우천)은 별도 기록으로 관리하며, **모든 사람은 클래스 0 person으로 표시**합니다. **사람이 없는 프레임**도 일부 포함하고 빈 라벨 파일을 만듭니다. 인도 위의 사람을 포함해 **보이는 모든 사람**에게 라벨을 작성합니다. 경계 상자는 횡단보도 관심 영역이 아니라 화면에 보이는 사람의 범위를 둘러싸야 합니다. CVAT 등의 라벨 작성 도구를 사용하고 **YOLO 객체 감지 형식**으로 내보냅니다. 야간과 우천은 사람의 클래스가 아닌 *촬영 조건*입니다.

예시:

```powershell
python scripts/extract_frames.py --video data/raw/night/night01.mp4 --out data/staging/night01 --every-sec 2
python scripts/extract_frames.py --video data/raw/rain/rain01.mp4 --out data/staging/rain01 --every-sec 2
```

라벨 작성 후 이미지와 확장자를 제외한 이름이 같은 `.txt` 라벨을 다음 위치에 저장합니다.

```
data/pedestrian/images/{train,val,test}/image01.jpg
data/pedestrian/labels/{train,val,test}/image01.txt
```

객체 하나당 라벨 한 줄을 작성합니다. 다음은 형식을 설명하기 위한 표기이며, 실제 파일에는 다섯 개의 숫자를 입력합니다.

```text
0 중심_x 중심_y 너비 높이
```

좌표는 0~1 범위로 정규화하며, 이미지와 라벨 파일은 확장자를 제외한 이름이 같아야 합니다. data/pedestrian/data.yaml에서 사람 클래스 하나를 정의합니다. **라벨 작성이 끝나기 전에는 학습하지 마세요.** 사람이 없는 사진에도 빈 `.txt` 라벨 파일을 추가합니다.

## 4. 데이터 검사 및 학습

```powershell
python scripts/check_dataset.py
# CPU에서 소규모로 실행을 확인합니다.
python scripts/train.py --epochs 2 --batch 2 --device cpu --name smoke
# 이후 전체 학습을 진행합니다(NVIDIA CUDA GPU 사용).
python scripts/train.py --epochs 40 --batch 8 --device 0 --name weather_person_v1
```

기본값인 yolo11n.pt는 학습의 시작점으로 COCO 사전 학습 가중치를 내려받습니다. **추가 학습에는 팀이 직접 수집하고 라벨을 작성한 데이터셋을 사용**합니다. NVIDIA CUDA GPU가 없는 컴퓨터에서도 --device cpu로 실행할 수 있지만, 전체 학습은 오래 걸릴 수 있습니다. 학습된 최적 가중치는 runs/weather_person_v1/weights/best.pt에 저장됩니다. COCO의 80개 클래스 전체를 감지하려면 기본 가중치를 사용합니다. 사용자 데이터로 학습한 **단일 클래스** 모델은 person만 출력합니다.

```powershell
python scripts/evaluate.py --weights runs/weather_person_v1/weights/best.pt
```

시험용 데이터에는 학습에 사용하지 않은, 라벨이 있는 이미지가 있어야 합니다. 별도의 메타데이터 표나 보고서를 사용해 주간, 야간, 우천별 오류 사례를 평가합니다. 기본 모델과 추가 학습 모델은 **동일한 시험 이미지**로 비교합니다. COCO 기본 모델과 단일 클래스 사용자 모델은 클래스 번호 대응 관계가 다를 수 있으므로 지표 비교 설정을 확인합니다.

## 5. 카메라 실시간 분석

프로젝트의 `model` 폴더에서 실행합니다. 기본 입력은 내장 카메라(`0`)입니다.
사진을 찍는 방식이 아니라, 카메라 영상을 계속 받아 YOLO로 보행자를 감지·추적합니다.

```powershell
python run_demo.py
# USB 카메라: 장치 번호에 따라 1, 2 등으로 변경
python run_demo.py --source 1
# IP 카메라: 접근 권한이 있는 실제 스트림 주소 사용
python run_demo.py --source "rtsp://USERNAME:PASSWORD@HOST:554/stream" --roi-file config/ip_roi.json
```

VS Code 실행 및 디버그에서 **카메라 실시간 분석**을 선택할 수도 있습니다.
실행기는 필요한 패키지가 설치된 `.venv`, 활성 환경, 기존 `yolo_project` 환경 등을 찾습니다.
기본 모델은 `yolo11n.pt`이며, 추가 학습한 모델은 `--model 경로/best.pt`로 지정합니다.

- 처음 연결한 카메라는 **전체 화면에서 즉시 보행자 감지**를 시작합니다.
- 횡단보도 안에 있는지 구분하려면 **z**를 누르고 영역을 드래그한 뒤 Enter로 확정합니다.
  영역을 지정하는 동안에만 화면 한 장을 고정합니다. C로 취소하면 기존 분석을 이어갑니다.
- 카메라 영역은 `config/camera_0_roi.json`처럼 장치별로 저장하며, 영상 파일의 영역과 분리합니다.
  카메라 위치나 촬영 각도가 바뀌면 z로 다시 지정하세요. IP 카메라마다 다른 `--roi-file`을 사용하세요.
- **p**는 분석 일시 정지/재개, **q** 또는 창 닫기는 종료입니다.
  일시 정지 중에도 카메라를 계속 읽으므로 재개하면 현재 시점부터 분석합니다.
- **g / r / u**는 보행자 신호를 녹색 / 적색 / 알 수 없음으로 수동 설정합니다.

카메라 수신을 별도 스레드에서 계속 처리하고 **최신 프레임 하나만** 분석에 전달합니다.
분석이 카메라보다 느리면 중간 프레임을 건너뛰어 오래된 영상이 밀리지 않도록 합니다.
화면에는 분석 FPS와 **프레임 수신 후 분석까지 걸린 시간**을 표시합니다.
이 시간에는 카메라 내부나 네트워크에서 발생한 지연은 포함되지 않습니다.
실제 분석 속도는 컴퓨터와 모델에 따라 달라지며, 느릴 때는 분석 크기를 줄일 수 있습니다.

```powershell
python run_demo.py --imgsz 416
# NVIDIA GPU 사용 환경이 준비된 경우
python run_demo.py --device 0
# CPU로 실행
python run_demo.py --device cpu
```

새 프레임이 5초 이상 도착하지 않거나 입력 오류가 나면 오류를 표시하고 종료합니다.
시간은 `--live-timeout 10`처럼 조정할 수 있습니다. 연결을 복구한 뒤 다시 실행하세요.
Windows 카메라 앱이나 화상회의 앱이 장치를 사용 중이면 먼저 종료하세요.

```powershell
# 연결 확인만 수행 (실시간 분석은 하지 않음)
python run_demo.py --source 0 --check-camera
# 특정 Windows 연결 방식 선택
python run_demo.py --source 0 --camera-backend dshow
```

저장된 영상 분석도 사용할 수 있습니다. 영상 파일은 프레임을 순서대로 처리합니다.

```powershell
python run_demo.py --source data/raw/crosswalk.mp4
python run_demo.py --source data/raw/crosswalk.mp4 --output recorded.mp4
```

`--output`을 지정했을 때만 분석 화면을 동영상으로 저장합니다.
저장 FPS는 입력 영상의 FPS를 사용하므로, 실시간 분석 FPS가 낮으면 저장 영상의 재생 시간은 실제 경과 시간보다 짧아질 수 있습니다.
Windows에서 카메라 접근이 차단되면 설정 > 개인정보 및 보안 > 카메라에서 데스크톱 앱 접근 허용을 확인하세요.
다른 운영체제에서는 한글 표시를 위해 나눔고딕, Noto Sans CJK 또는 AppleGothic이 필요합니다.
RTSP 접속 정보와 촬영 영상을 GitHub에 올리지 마세요.

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
