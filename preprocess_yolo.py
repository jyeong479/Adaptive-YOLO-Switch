import os
import json
import shutil
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 영역
# ---------------------------------------------------------
DATA_ROOT = './101.교통문제_해결을_위한_CCTV_교통_데이터(시내도로)/01.데이터/1.Training'

# 새롭게 저장될 최상위 폴더명
YOLO_OUTPUT_DIR = './yolo_dataset101/train' 

# 표 2, 3번 작업: 클래스 통합 매핑 딕셔너리
CLASS_MAPPING = {
    1: 0,  # 승용차 -> 0
    2: 1,  # 소형버스 -> 1
    3: 1,  # 대형버스 -> 1
    4: 2,  # 트럭 -> 2
    5: 2,  # 대형 트레일러 -> 2
    6: 3,  # 오토바이 -> 3
    7: 4   # 보행자 -> 4
}

def build_image_index(root_dir):
    print("이미지 파일 위치를 스캔 중입니다. 잠시만 기다려주세요...")
    img_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_dict[f] = os.path.join(dirpath, f)
    print(f"총 {len(img_dict)}개의 이미지를 찾았습니다!\n")
    return img_dict

def get_condition_folder(time_str, weather):
    try:
        # "18:59:58" 에서 "18"만 추출
        hour = int(time_str.split(':')[0])
        
        # 아침 7시 ~ 저녁 6시(18시 59분)까지만 Day로 엄격하게 판별
        if 7 <= hour <= 18:
            day_night = 'day'
        else:
            day_night = 'night'
    except:
        day_night = 'day' # 시간 정보가 이상할 경우 기본값
        
    if weather.lower() == 'sunny':
        weather_cond = 'clear'
    else:
        weather_cond = 'unclear'
        
    return f"{day_night}_{weather_cond}"

def process_datasets():
    image_path_index = build_image_index(DATA_ROOT)

    json_files = []
    for dirpath, _, filenames in os.walk(DATA_ROOT):
        for f in filenames:
            if f.endswith('.json'):
                json_files.append(os.path.join(dirpath, f))

    for json_path in json_files:
        print(f"처리 중: {os.path.basename(json_path)}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 ID로 [파일명, meta_id]를 찾기 위한 딕셔너리 생성
        img_info_dict = {
            img['id']: {'file_name': os.path.basename(img['file_name']), 'meta_id': img.get('meta_id')} 
            for img in data.get('images', [])
        }
        
        # meta_id로 [시간대, 날씨]를 찾기 위한 딕셔너리 생성
        meta_dict = {
            m['id']: {'time': m.get('time', '12:00:00'), 'weather': m.get('weather', '')}
            for m in data.get('meta', [])
        }

        # annotations 분석 시작
        for anno in data.get('annotations', []):
            img_id = anno.get('image_id')
            bboxes = anno.get('bbox', [])
            categories = anno.get('category_id', [])
            
            if not bboxes or not categories:
                continue

            # 해당 어노테이션의 이미지 정보 및 메타 정보 가져오기
            img_info = img_info_dict.get(img_id)
            if not img_info:
                continue
                
            json_file_name = img_info['file_name']
            meta_id = img_info['meta_id']
            
            # 메타데이터를 기반으로 저장될 하위 폴더 이름 결정 (예: day_clear)
            meta_info = meta_dict.get(meta_id, {'time': '12:00:00', 'weather': 'Unknown'})
            condition_folder_name = get_condition_folder(meta_info['time'], meta_info['weather'])
            
            # 실제 컴퓨터에 존재하는 이미지 경로 찾기
            real_img_path = image_path_index.get(json_file_name)
            if not real_img_path:
                continue

            # 이미지 해상도 가져오기
            try:
                with Image.open(real_img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                continue

            valid_yolo_lines = []

            for bbox, cat_id in zip(bboxes, categories):
                if cat_id not in CLASS_MAPPING:
                    continue
                
                yolo_class = CLASS_MAPPING[cat_id]
                x_min, y_min, x_max, y_max = bbox

                if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
                    continue 
                if x_max <= x_min or y_max <= y_min:
                    continue 

                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                w = (x_max - x_min) / img_width
                h = (y_max - y_min) / img_height

                valid_yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 유효한 객체가 있을 경우 조건별 폴더에 저장
            if valid_yolo_lines:
                base_name = os.path.splitext(json_file_name)[0]
                
                # 조건별 폴더 경로 동적 생성 (예: yolo_dataset_v2/train/day_clear/images)
                target_img_dir = os.path.join(YOLO_OUTPUT_DIR, condition_folder_name, 'images')
                target_label_dir = os.path.join(YOLO_OUTPUT_DIR, condition_folder_name, 'labels')
                
                os.makedirs(target_img_dir, exist_ok=True)
                os.makedirs(target_label_dir, exist_ok=True)
                
                new_img_path = os.path.join(target_img_dir, json_file_name)
                new_label_path = os.path.join(target_label_dir, f"{base_name}.txt")

                with open(new_label_path, 'w', encoding='utf-8') as lf:
                    lf.write('\n'.join(valid_yolo_lines))
                
                if not os.path.exists(new_img_path):
                    shutil.copy2(real_img_path, new_img_path)

    print("\n모든 전처리, YOLO 변환 및 [환경별 폴더 분류] 작업 완료")

if __name__ == "__main__":
    process_datasets()