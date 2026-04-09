import os
import random
import shutil
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 공식 저장소의 핵심 클래스 가져오기
from src.cyclegan_turbo import CycleGAN_Turbo

# =====================================================================
# 경로 설정 (현재 서버의 실제 데이터 경로로 수정하세요)
# =====================================================================
SRC_DAY_CLEAR = '../YOLO/dataset_merged/train/day_clear'
SRC_DAY_UNCLEAR = '../YOLO/dataset_merged/train/day_unclear'

OUT_DAY_UNCLEAR = '../YOLO/dataset_merged/train/day_unclear_turbo'
OUT_NIGHT_UNCLEAR = '../YOLO/dataset_merged/train/night_unclear_turbo'

# 생성 수량 목표
TARGET_DAY_UNCLEAR = 13808
TARGET_NIGHT_UNCLEAR = 25510

# =====================================================================

def setup_dirs(out_dir):
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)

def tensor_to_pil(tensor):
    """모델의 Tensor 출력을 다시 이미지로 변환"""
    out_arr = tensor.squeeze(0).cpu().float().numpy()
    out_arr = (out_arr * 0.5 + 0.5) * 255.0 # -1~1 -> 0~255
    out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.transpose(out_arr, (1, 2, 0)))

def run_turbo_batch(task_name, src_base_dir, out_base_dir, target_count, pretrained_model_name):
    print(f"\n🚀 [{task_name}] 모델 로드 중: {pretrained_model_name}")
    
    # 1. 모델 로드 (Hugging Face에서 가중치가 자동 다운로드 됨)
    model = CycleGAN_Turbo(pretrained_name=pretrained_model_name)
    model.eval()
    
    setup_dirs(out_base_dir)
    
    # 전처리: SD-Turbo는 512x512 해상도 최적화
    T = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    src_img_dir = os.path.join(src_base_dir, 'images')
    src_lbl_dir = os.path.join(src_base_dir, 'labels')
    
    all_imgs = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected = random.sample(all_imgs, min(target_count, len(all_imgs)))
    
    print(f"작업 시작: 총 {len(selected)}장 변환")
    
    with torch.no_grad():
        for idx, img_name in enumerate(selected):
            img_path = os.path.join(src_img_dir, img_name)
            file_name, ext = os.path.splitext(img_name)
            lbl_path = os.path.join(src_lbl_dir, file_name + '.txt')
            
            # 원본 이미지 읽기 (해상도 저장)
            img_pil = Image.open(img_path).convert('RGB')
            orig_size = img_pil.size # (width, height)
            
            # 전처리 및 추론
            img_tensor = T(img_pil).unsqueeze(0).cuda()
            out_tensor = model(img_tensor) # 변환 수행
            
            # 후처리 및 원본 비율로 복원
            # (바운딩 박스가 엇나가는 것을 방지하기 위해 원래 해상도로 되돌림)
            out_pil = tensor_to_pil(out_tensor)
            out_pil = out_pil.resize(orig_size, Image.LANCZOS)
            
            # 저장
            out_img_path = os.path.join(out_base_dir, 'images', f"{file_name}_turbo{ext}")
            out_pil.save(out_img_path)
            
            # 라벨 복사
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, os.path.join(out_base_dir, 'labels', f"{file_name}_turbo.txt"))
                
            if (idx + 1) % 500 == 0:
                print(f" -> 진행 상황: {idx + 1} / {len(selected)}")
                
    # 메모리 초기화 (다음 모델을 위해 VRAM 비우기)
    del model
    torch.cuda.empty_cache()
    print(f"✅ [{task_name}] 완료!\n")

if __name__ == '__main__':
    # 1. 맑은 날(DayClear) -> 비 오는 날(DayUnclear) 변환
    # 공식 모델: clear_to_rainy 사용
    run_turbo_batch(
        task_name="Day -> DayUnclear",
        src_base_dir=SRC_DAY_CLEAR,
        out_base_dir=OUT_DAY_UNCLEAR,
        target_count=TARGET_DAY_UNCLEAR,
        pretrained_model_name="clear_to_rainy"
    )
    
    # 2. 흐린 날(DayUnclear 원본) -> 비/흐린 밤(NightUnclear) 변환
    # 공식 모델: day_to_night 사용
    run_turbo_batch(
        task_name="DayUnclear -> NightUnclear",
        src_base_dir=SRC_DAY_UNCLEAR,
        out_base_dir=OUT_NIGHT_UNCLEAR,
        target_count=TARGET_NIGHT_UNCLEAR,
        pretrained_model_name="day_to_night"
    )