# 🚀 YOLO Adaptive Switching Study: Data Pipeline

본 저장소는 환경 변화(시간, 날씨)에 따른 **YOLO 모델 스위칭 전략** 연구를 위한 데이터 전처리 및 증강 파이프라인을 포함함

## 📋 연구 프로세스 (Research Workflow)

### 1️⃣ 데이터 추출 (Extraction)
* `unzip_data.py`: 원천 데이터셋 압축 해제 및 초기 디렉토리 구조 생성

---

### 2️⃣ 데이터 정제 및 통합 (Pre-processing)
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

### 3️⃣ 데이터셋 병합 (Merging)
* `merge_dataset.py`: 전처리 완료된 이종 데이터셋을 단일 통합 학습 셋으로 병합

---

### 4️⃣ 데이터 증강 (Augmentation)
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
