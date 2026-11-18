# 학습 데이터 출처

원본: 저장소 루트의 pedestrian crosswalk detection.v2i.yolov11
Roboflow: https://universe.roboflow.com/project-onkj7/pedestrian-crosswalk-detection/dataset/2
라이선스: CC BY 4.0. 원본 README.dataset.txt와 README.roboflow.txt를 함께 보존했습니다.

원본 train/valid/test를 images/{train,val,test}, labels/{train,val,test}로 복사했습니다.
train 43쌍, val 12쌍, test 6쌍이며 원본 라벨 내용과 클래스 번호(0 bicycle, 1 humans)는 유지합니다.
원본 폴더는 보존합니다.

61장의 소규모 데이터이며 파일명상 같은 영상의 프레임이 여러 분할에 포함됩니다.
최종 성능은 독립된 촬영 영상과 주간/야간/우천별 데이터로 별도 평가해야 합니다.
