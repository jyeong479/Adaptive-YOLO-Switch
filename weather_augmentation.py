import os
import cv2
import numpy as np
import random
import shutil

# =====================================================================
# [설정 1] 경로 지정 (작업 환경에 맞게 폴더 경로를 수정하세요)
# =====================================================================
# 1. DayClear -> DayUnclear 생성용 경로
SRC_DAY_CLEAR_IMG = 'YOLO/dataset_merged/train/day_clear/images'
SRC_DAY_CLEAR_LBL = 'YOLO/dataset_merged/train/day_clear/labels'

OUT_DAY_UNCLEAR_IMG = 'YOLO/dataset_merged/train/day_unclear_aug/images'
OUT_DAY_UNCLEAR_LBL = 'YOLO/dataset_merged/train/day_unclear_aug/labels'

# 2. DayUnclear -> NightUnclear 생성용 경로 (원본 DayUnclear를 밤으로 바꿈)
SRC_DAY_UNCLEAR_IMG = 'YOLO/dataset_merged/train/day_unclear/images'
SRC_DAY_UNCLEAR_LBL = 'YOLO/dataset_merged/train/day_unclear/labels'

OUT_NIGHT_UNCLEAR_IMG = 'YOLO/dataset_merged/train/night_unclear_aug/images'
OUT_NIGHT_UNCLEAR_LBL = 'YOLO/dataset_merged/train/night_unclear_aug/labels'

# 샘플 확인용 폴더
SAMPLE_DIR = 'YOLO/dataset_merged/train/samples_aug'

# =====================================================================
# [설정 2] 135,000장 기준 증강 목표 수량
# =====================================================================
TARGET_DAY_UNCLEAR = 41423  # DayClear에서 추출하여 변환
TARGET_NIGHT_UNCLEAR = 76532 # DayUnclear 원본에서 추출하여 변환

# =====================================================================

def create_dirs():
    dirs = [OUT_DAY_UNCLEAR_IMG, OUT_DAY_UNCLEAR_LBL, 
            OUT_NIGHT_UNCLEAR_IMG, OUT_NIGHT_UNCLEAR_LBL, SAMPLE_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def apply_unclear_weather(image):
    """맑은 주간(DayClear)을 흐리고 안개 낀 주간(DayUnclear)으로 변환"""
    # 1. 대비(Contrast) 감소 - 흐린 날씨의 밋밋한 색감 표현
    img_float = image.astype(np.float32) / 255.0
    img_float = (img_float - 0.5) * 0.6 + 0.5 
    img_float = np.clip(img_float, 0, 1)
    
    # 2. 안개(Fog) 효과 덮어씌우기
    fog_layer = np.ones_like(img_float) * 0.85 # 밝은 회색
    blended = cv2.addWeighted(img_float, 0.75, fog_layer, 0.25, 0)
    
    # 3. 블러(가우시안) 추가 - 시야가 흐려진 느낌
    blurred = cv2.GaussianBlur(blended, (5, 5), 0)
    
    return (blurred * 255).astype(np.uint8)

def apply_night_filter(image):
    """흐린 주간(DayUnclear)을 어두운 야간(NightUnclear)으로 변환"""
    # 1. 감마 교정 (어둡게 만들기)
    gamma = 2.8 
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    darkened = cv2.LUT(image, table)
    
    # 2. HSV 조절 (채도와 명도를 더 낮춰서 깊은 밤 느낌)
    hsv = cv2.cvtColor(darkened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 0.6, 0, 255).astype('uint8')
    v = np.clip(v * 0.4, 0, 255).astype('uint8')
    
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def draw_bboxes(image, label_path):
    """바운딩 박스 그리기 (샘플 확인용)"""
    drawn_img = image.copy()
    if not os.path.exists(label_path): return drawn_img
    
    img_h, img_w = drawn_img.shape[:2]
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id, cx, cy, w, h = map(float, parts[:5])
            
            xmin = int((cx - w / 2) * img_w)
            ymin = int((cy - h / 2) * img_h)
            xmax = int((cx + w / 2) * img_w)
            ymax = int((cy + h / 2) * img_h)
            
            cv2.rectangle(drawn_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(drawn_img, str(int(cls_id)), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return drawn_img

def process_augmentation(task_name, src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, target_count, aug_func):
    """지정된 수량만큼 증강을 수행하는 핵심 함수"""
    print(f"\n[시작] {task_name} 증강 (목표: {target_count}장)")
    
    all_imgs = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(all_imgs, min(target_count, len(all_imgs)))
    
    for idx, img_name in enumerate(selected):
        img_path = os.path.join(src_img_dir, img_name)
        file_name, ext = os.path.splitext(img_name)
        lbl_path = os.path.join(src_lbl_dir, file_name + '.txt')
        
        # 원본 읽기
        image = cv2.imread(img_path)
        if image is None: continue
            
        # 변환 함수 적용 (안개 추가 or 야간화)
        aug_image = aug_func(image)
        
        # 파일 저장
        new_name = f"{file_name}_aug{ext}"
        cv2.imwrite(os.path.join(out_img_dir, new_name), aug_image)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(out_lbl_dir, f"{file_name}_aug.txt"))
            
        # 초반 30장 전/후 비교 샘플 저장
        if idx < 30:
            sample_before = draw_bboxes(image, lbl_path)
            sample_after = draw_bboxes(aug_image, lbl_path)
            compare_img = cv2.hconcat([sample_before, sample_after])
            sample_name = f"{task_name}_{file_name}_compare{ext}"
            cv2.imwrite(os.path.join(SAMPLE_DIR, sample_name), compare_img)
            
        if (idx + 1) % 5000 == 0:
            print(f" -> 진행 중: {idx + 1} / {len(selected)} 완료")
            
    print(f"[완료] {task_name} 처리 끝\n")

def main():
    create_dirs()
    
    # 1. DayClear -> DayUnclear (흐리고 안개 낀 날씨 생성)
    process_augmentation(
        task_name="Day -> DayUnclear",
        src_img_dir=SRC_DAY_CLEAR_IMG, src_lbl_dir=SRC_DAY_CLEAR_LBL,
        out_img_dir=OUT_DAY_UNCLEAR_IMG, out_lbl_dir=OUT_DAY_UNCLEAR_LBL,
        target_count=TARGET_DAY_UNCLEAR,
        aug_func=apply_unclear_weather
    )
    
    # 2. DayUnclear(원본) -> NightUnclear (이미 흐린 날씨를 밤으로 만듦)
    process_augmentation(
        task_name="DayUnclear -> NightUnclear",
        src_img_dir=SRC_DAY_UNCLEAR_IMG, src_lbl_dir=SRC_DAY_UNCLEAR_LBL,
        out_img_dir=OUT_NIGHT_UNCLEAR_IMG, out_lbl_dir=OUT_NIGHT_UNCLEAR_LBL,
        target_count=TARGET_NIGHT_UNCLEAR,
        aug_func=apply_night_filter
    )
    
    print("모든 증강 작업 및 샘플 생성이 완료")

if __name__ == '__main__':
    main()