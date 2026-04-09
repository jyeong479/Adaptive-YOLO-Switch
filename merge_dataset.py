import os
import shutil

# 경로 설정 (현재 위치에 맞게 수정하세요)
SRC_DIR = 'dataset101'
DST_DIR = 'dataset'

moved_count = 0
renamed_count = 0

print("데이터 병합을 시작")

# dataset101 안의 모든 폴더와 파일을 재귀적으로 탐색
for root, dirs, files in os.walk(SRC_DIR):
    for file in files:
        # 1. 원본 파일의 절대 경로
        src_file_path = os.path.join(root, file)
        
        # 2. 이동할 대상 폴더 경로 계산 
        # (예: dataset101/train/day_clear/images -> dataset/train/day_clear/images)
        relative_path = os.path.relpath(root, SRC_DIR)
        dst_folder_path = os.path.join(DST_DIR, relative_path)
        
        # 대상 폴더가 없다면 생성
        os.makedirs(dst_folder_path, exist_ok=True)
        
        # 3. 대상 파일 경로 설정
        dst_file_path = os.path.join(dst_folder_path, file)
        
        # 4. 파일 이름 중복 검사 및 덮어쓰기 방지 처리
        if os.path.exists(dst_file_path):
            file_name, ext = os.path.splitext(file)
            # 이름이 겹치면 뒤에 _101을 붙여서 저장
            new_file_name = f"{file_name}_101{ext}"
            dst_file_path = os.path.join(dst_folder_path, new_file_name)
            renamed_count += 1
            
        # 5. 파일 이동 (이동이 아닌 복사를 원하면 shutil.move 대신 shutil.copy2 사용)
        shutil.move(src_file_path, dst_file_path)
        moved_count += 1
        
        if moved_count % 10000 == 0:
            print(f" -> 진행 중: {moved_count}개 파일 이동 완료...")

print("\n데이터셋 병합이 완료")
print(f"총 이동된 파일 수: {moved_count}개")
print(f"이름이 중복되어 변경된 파일 수: {renamed_count}개")