# 🚀 YOLO Adaptive Switching Study: Data Pipeline

본 저장소는 환경 변화(시간, 날씨)에 따른 **YOLO 모델 스위칭 전략** 연구를 위한 데이터 전처리 및 증강 파이프라인을 포함함


## 🧐 연구 배경 및 목적 (Motivation & Objective)

### 1. 연구 배경 및 필요성
* **ITS 환경의 가변성**: 지능형 교통 시스템(ITS)은 조도 변화와 기상 악화 등 가변적인 환경에서도 일관된 탐지 성능을 유지해야 함
* **도메인 격차(Domain Gap) 발생**: 기상 요인은 딥러닝 모델의 출력 값에 비선형적 영향을 미치며, 학습 데이터와 실제 환경 간의 격차를 유발함
* **단일 모델의 한계**: 특정 환경에 특화된 학습은 이질적인 도메인에서 급격한 성능 저하(Domain Shift)를 야기하여 시스템의 신뢰성을 저해함

### 2. 연구 목적 및 차별성
* **구조적 강건성 비교 분석**: 앵커 기반(YOLOv5)과 앵커 프리(YOLOv8) 구조의 차이가 환경적 불확실성 극복에 미치는 영향을 정량적으로 분석함
* **8×8 교차 평가 매트릭스 설계**: 8개의 세부 도메인으로 격리된 환경에서 총 64개의 시나리오를 통해 도메인 전이 패턴을 규명함
* **모델 스위칭(Model Switching) 전략 제안**: 단일 모델 기반 접근의 한계를 극복하기 위해 도로 상황에 따라 최적의 모델을 선택적으로 적용하는 전략의 가이드라인을 제공함



### 3. 본 파이프라인의 학술적 역할
* **체계적인 도메인 구조화**: 조도(주간/야간) 및 기상(맑음/비맑음) 조건의 조합을 통해 환경적 이질성을 반영한 데이터셋을 세분화함
* **AI 기반 데이터 증강(CycleGAN-Turbo)**: 확보가 어려운 악천후 및 야간 데이터를 인위적으로 생성하여 데이터 불균형 문제를 해결하고 모델의 일반화 능력을 고도화함
* **데이터 무결성 확보**: 원천 데이터의 정규화 및 YOLO 표준 형식 변환을 통해 연구의 재현성을 보장하는 고품질 학습 데이터를 구축함



## 📂 데이터셋 (Datasets)
본 연구에는 **AI 허브(AI Hub)** 에서 제공하는 국내 도로 환경 데이터셋을 사용함

* **[시내도로]** [교통문제 해결을 위한 CCTV 교통 영상(시내도로)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=165)
* **[고속도로]** [교통문제 해결을 위한 CCTV 교통 영상(고속도로)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=164)

## 📋 연구 프로세스 (Research Workflow)

### 1️ 데이터 추출 (Extraction)
* `unzip_data.py`: 원천 데이터셋 압축 해제 및 초기 디렉토리 구조 생성

---

### 2️ 데이터 정제 및 통합 (Pre-processing)
`preprocess_yolo.py`, `preprocess_yolo_highway.py`를 통한 데이터 무결성 확보 및 라벨 체계 통합

* **데이터 클리닝**:
    * 유효 범위 외 바운딩 박스(BB) 좌표 보유 데이터 제거
    * 객체 미존재(No-object) 이미지 선별 및 제외
* **포맷 변환**: YOLO 학습 규격(`.txt`)에 따른 라벨링 정보 일괄 변환
* **클래스 통합**: 시내도로 및 고속도로 데이터셋의 이질적 클래스 구성을 표준 체계로 단일화

| Original Category (시내/고속) | Standardized Class | ID |
| :--- | :--- | :---: |
| 승용차 (Car) | **car** | 0 |
| 소형/대형 버스 (Bus) | **bus** | 1 |
| 트럭 / 대형 트레일러 (Truck) | **truck** | 2 |
| 오토바이 (Motorcycle) | **bike** | 3 |
| 보행자 (Pedestrian) | **person** | 4 |

---

### 3️ 데이터셋 병합 (Merging)
* `merge_dataset.py`: 전처리 완료된 이종 데이터셋을 단일 통합 학습 셋으로 병합

---

### 4️ 데이터 증강 (Augmentation)
환경 변화 대응력 강화를 위한 2단계 증강 전략 수행

#### **A. 광학적 증강 (Light Augmentation)**
* `weather_augmentation.py`: 밝기(Brightness), 명암(Contrast), 채도(Saturation) 조절을 통한 기초 환경 변이 모사

#### **B. 생성 모델 기반 증강 (AI Augmentation)**
* `batch_augment.py`: **CycleGAN-Turbo (Clear-to-Rainy)** 활용, 기상 상황(맑음→우천) 도메인 변환 수행 및 모델 강건성(Robustness) 확보

---

## 🛠 기술 스택
- **Model**: YOLO Series
- **Augmentation**: CycleGAN-Turbo, OpenCV
- **Language**: Python
