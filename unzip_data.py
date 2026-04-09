import os
import zipfile

def extract_nested_zips(root_path):
    """
    최상위 폴더부터 하위 폴더까지 모두 탐색하여 
    .zip 파일을 찾고, 해당 위치에 압축을 해제합니다.
    """
    # os.walk로 최상위 경로 하위의 모든 폴더와 파일을 순회
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.zip'):
                zip_file_path = os.path.join(dirpath, filename)
                
                # 압축을 풀 폴더 이름 설정 (예: 갑천교네거리.zip -> 갑천교네거리 폴더)
                folder_name = filename.replace('.zip', '')
                extract_path = os.path.join(dirpath, folder_name)
                
                # 이미 압축을 푼 폴더가 없다면 생성 후 압축 해제 진행
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path, exist_ok=True)
                    print(f"압축 해제 진행 중: {zip_file_path}")
                    
                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_path)
                    except zipfile.BadZipFile:
                        print(f"경고: 파일이 손상되었습니다 - {zip_file_path}")
                    except Exception as e:
                        print(f"에러 발생 ({filename}): {e}")
                else:
                    print(f"건너뜀 (이미 압축 해제됨): {extract_path}")

    print("\n모든 탐색 및 압축 해제 작업이 완료되었습니다!")

# ---------------------------------------------------------
# 실행 부분: 실제 데이터가 있는 최상위 폴더의 경로를 입력하세요.
# (101.교통문제_.../01.데이터 폴더의 절대 경로나 상대 경로)
# ---------------------------------------------------------
target_directory = './101.교통문제_해결을_위한_CCTV_교통_데이터(시내도로)/01.데이터'

extract_nested_zips(target_directory)