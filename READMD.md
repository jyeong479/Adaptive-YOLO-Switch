# YOLO 연구 순서

## 1. unzip_data.py (압축 해제)

## 2. preprocess_yolo.py, preprocess_yolo_highway.py(증강 제외한 전처리)
- 유효하지 않은 바운딩 박스 좌표 값을 레이블로 갖는 데이터 선별
- 데이터 세트 간의 클래스 분류 통합 및 값의 재정의로 인한 라벨링 데이터 일괄 수정
    - 국내 시내도로 데이터
    승용차: 1 -> car: 0
    소형버스: 2 -> bus: 1
    대형버스: 3 -> bus: 1
    트럭: 4 -> truck: 2
    대형 트레일러: 5 -> truck: 2
    오토바이: 6 -> bike: 3
    보행자: 7 -> person: 4
    - 국내 고속도로 데이터
    car: 0
    bus:1
    truck: 2
    bike: 3
    person: 4
- 영상 내 객체가 존재하지 않는 데이터 선별
- 데이터 디렉토리 구성을 학습 및 추론에 용이하도록 일괄적으로 변경
- 객체 검출 모델에서 요구하는 바운딩 박스 정보 형식으로 Labeling 파일의 내용을 변환

## 3. merge_dataset.py (별도의 데이터셋 병합)

## 4. weather_augmentation.py(약한 증강)
- 밝기 ,명암, 채도 등을 조절하여 증강

## 5. batch_augment.py(cycleGAN-Turbo 이용한 증강)
- cycleGan-Turbo (Clear to Rainy) 활용하여 증강
