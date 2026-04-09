import os
import json
import shutil
from PIL import Image

# ---------------------------------------------------------
# 1. 설정 영역
# ---------------------------------------------------------
DATA_ROOT = './101.교통문제_해결을_위한_CCTV_교통_데이터(시내도로)/01.데이터/2.Validation'

# 새롭게 저장될 최상위 폴더명
YOLO_OUTPUT_DIR = './yolo_dataset101/val' 

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

def get_condition_folder(date_str, time_str, weather):
    # 월별 주간(Day) 기준 시간 설정: (시, 분)
    month_day_times = {
        1:  ((7, 45), (17, 35)),
        2:  ((7, 20), (18, 10)),
        3:  ((6, 40), (18, 40)),
        4:  ((6, 0),  (19, 10)),
        5:  ((5, 20), (19, 35)),
        6:  ((5, 5),  (19, 50)), # 가장 해가 긴 시기
        7:  ((5, 20), (19, 50)),
        8:  ((5, 45), (19, 25)),
        9:  ((6, 10), (18, 45)),
        10: ((6, 40), (18, 0)),
        11: ((7, 15), (17, 20)),
        12: ((7, 40), (17, 15))
    }

    try:
        # "2020-10-14" 에서 월(10) 추출
        month = int(date_str.split('-')[1])
        
        # "18:59:58" 에서 시간과 분 추출
        time_parts = time_str.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        
        # 분 단위로 환산하여 계산 (비교를 용이하게 하기 위함)
        current_minutes = hour * 60 + minute
        
        # 해당 월의 일출/일몰 시간 가져오기 (예외 시 기본값 7시~18시)
        sunrise, sunset = month_day_times.get(month, ((7, 0), (18, 0)))
        sunrise_minutes = sunrise[0] * 60 + sunrise[1]
        sunset_minutes = sunset[0] * 60 + sunset[1]
        
        # 월별 일출 일몰 시간에 따라 Day / Night 판별
        if sunrise_minutes <= current_minutes <= sunset_minutes:
            day_night = 'day'
        else:
            day_night = 'night'
    except:
        day_night = 'day' # 날짜/시간 정보 파싱 실패 시 기본값
        
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
        
        # meta_id로 [날짜, 시간대, 날씨]를 찾기 위한 딕셔너리 생성 (date 추가됨)
        meta_dict = {
            m['id']: {
                'date': m.get('date', '2020-01-01'), 
                'time': m.get('time', '12:00:00'), 
                'weather': m.get('weather', '')
            }
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
            meta_info = meta_dict.get(meta_id, {'date': '2020-01-01', 'time': '12:00:00', 'weather': 'Unknown'})
            
            # date, time, weather를 모두 넘겨주도록 변경
            condition_folder_name = get_condition_folder(meta_info['date'], meta_info['time'], meta_info['weather'])
            
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