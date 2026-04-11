import cv2
import numpy as np
import os
import pandas as pd
import glob
from ultralytics import YOLO
from tqdm import tqdm
import gc

# =========================================================
# [1] 설정
# =========================================================
FOLDER_PATH = r"G:\내 드라이브\졸논\데이터셋\VIRAT" 
RESULT_DIR = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\hyewon\agent\result\0410"
FRAMES_SAVE_DIR = os.path.join(RESULT_DIR, "selected_frames_alpha_01")
CSV_PATH = os.path.join(RESULT_DIR, "experiment_results_0410.csv")

MODEL = YOLO('yolov8n.pt')
CALIB_FRAMES = 200
TARGET_ALPHA = 0.1
TARGET_K = 1.28
LQ_SAMPLING_RATE = 10  # 10프레임마다 1번씩만 LQ 분석 수행

# =========================================================
# [2] 유틸리티
# =========================================================
def get_total_frames(v_path):
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened(): return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def save_to_csv(data_list):
    if not data_list: return
    os.makedirs(RESULT_DIR, exist_ok=True)
    df_new = pd.DataFrame(data_list)
    if not os.path.exists(CSV_PATH):
        df_new.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(CSV_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')

# =========================================================
# [3] 메인 실험
# =========================================================
def run_full_experiment():
    video_files = glob.glob(os.path.join(FOLDER_PATH, "*.mp4"))
    if not video_files:
        print(f"[Error] 비디오 파일 없음: {FOLDER_PATH}")
        return

    os.makedirs(FRAMES_SAVE_DIR, exist_ok=True)
    print(f"🚀 총 {len(video_files)}개 비디오 분석 시작 (LQ 1/10 샘플링 및 캐싱 적용)\n")

    for v_idx, v_path in enumerate(video_files):
        video_name = os.path.basename(v_path)
        total_frames_est = get_total_frames(v_path)
        if total_frames_est == 0:
            print(f"Skipping {video_name}")
            continue

        print(f"\n[{v_idx+1}/{len(video_files)}] {video_name} ({total_frames_est}프레임)")

        # -------------------------------------------------------
        # [1단계] Calibration (초기 학습)
        # -------------------------------------------------------
        cap = cv2.VideoCapture(v_path)
        calib_scores = []
        for _ in range(CALIB_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            res = MODEL(frame, verbose=False)
            if len(res[0].boxes) > 0:
                calib_scores.append(res[0].boxes.conf.max().item())
            del frame
        
        mu, sigma = (np.mean(calib_scores), np.std(calib_scores)) if calib_scores else (0.5, 0.1)
        # Alpha 0.1 기준의 임계값 (실시간 이미지 저장용)
        threshold_final = max(0.1, min(mu - (TARGET_K * sigma), 0.9))
        cap.release()
        
        print(f"    → Calibration 완료: μ={mu:.3f}, σ={sigma:.3f} | Alpha 0.1 Threshold={threshold_final:.3f}")

        # -------------------------------------------------------
        # [2단계] 메인 분석 (실시간 저장 및 LQ 캐싱 분석)
        # -------------------------------------------------------
        video_frame_dir = os.path.join(FRAMES_SAVE_DIR, video_name.replace('.mp4', ''))
        os.makedirs(video_frame_dir, exist_ok=True)

        cap = cv2.VideoCapture(v_path)
        all_max_conf = []        # Proposed 결정 및 전체 통계용
        all_has_det_hq = []      # Baseline & Proposed 정확도용
        all_has_det_lq = []      # Uniform(Q20) 정확도용
        all_buf_hq_size = []     # 용량 데이터 (High)
        all_buf_lq_size = []     # 용량 데이터 (Low)
        saved_count = 0

        with tqdm(total=total_frames_est, desc="  [처리중]", unit="f", ncols=80) as pbar:
            frame_idx = 0
            last_has_det_lq = False  # 샘플링 사이의 값을 채워줄 캐시 변수

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 1. 고화질(HQ) 분석 (매 프레임)
                res_hq = MODEL(frame, verbose=False)
                has_det_hq = len(res_hq[0].boxes) > 0
                max_c_hq = res_hq[0].boxes.conf.max().item() if has_det_hq else 0.0

                # 2. 전송 데이터량 측정을 위한 인코딩
                _, buf_hq = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                _, buf_lq = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

                # 3. 저화질(LQ) 분석 (10프레임 샘플링 및 캐싱)
                if frame_idx % LQ_SAMPLING_RATE == 0:
                    img_lq = cv2.imdecode(buf_lq, cv2.IMREAD_COLOR)
                    res_lq = MODEL(img_lq, verbose=False)
                    last_has_det_lq = len(res_lq[0].boxes) > 0  # 새로운 LQ 결과 업데이트
                    del img_lq
                
                # 샘플링 주기 사이에는 마지막으로 확인된 LQ 결과를 사용
                has_det_lq = last_has_det_lq 

                # 데이터 기록 (메모리 최적화를 위해 수치만 저장)
                all_max_conf.append(max_c_hq)
                all_has_det_hq.append(has_det_hq)
                all_has_det_lq.append(has_det_lq)
                all_buf_hq_size.append(len(buf_hq))
                all_buf_lq_size.append(len(buf_lq))

                # 4. 실시간 이미지 저장 (Alpha 0.1 기준 Proposed 프레임)
                if max_c_hq >= threshold_final:
                    img_path = os.path.join(video_frame_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    saved_count += 1

                # 5. 메모리 강제 해제
                del frame, buf_hq, buf_lq
                frame_idx += 1
                pbar.update(1)
                
                if frame_idx % 500 == 0: gc.collect()

        cap.release()

        # -------------------------------------------------------
        # [3단계] CSV 결과 산출 및 저장
        # -------------------------------------------------------
        total_frames = len(all_max_conf)
        baseline_det_frames = sum(all_has_det_hq)
        video_temp_results = []

        # (1) Baseline: 전 구간 고화질 전송
        video_temp_results.append({
            'Case': 'Baseline', 'Video': video_name, 'Alpha': '-', 'Threshold': '-',
            'Size_MB': round(sum(all_buf_hq_size)/(1024*1024), 2),
            'Total_Frames': total_frames, 'Sent_Frames': total_frames,
            'Det_Frames': baseline_det_frames, 'Baseline_Det_Frames': baseline_det_frames
        })

        # (2) Uniform(Q20): 전 구간 저화질 전송 (캐싱된 정확도 반영)
        video_temp_results.append({
            'Case': 'Uniform(Q20)', 'Video': video_name, 'Alpha': '-', 'Threshold': '-',
            'Size_MB': round(sum(all_buf_lq_size)/(1024*1024), 2),
            'Total_Frames': total_frames, 'Sent_Frames': total_frames,
            'Det_Frames': sum(all_has_det_lq), 'Baseline_Det_Frames': baseline_det_frames
        })

        # (3) Proposed: 다양한 Alpha 값에 따른 결과 계산
        alphas = {0.01: 2.33, 0.05: 1.645, 0.1: 1.28, 0.2: 0.84}
        for a_val, k_val in alphas.items():
            th = max(0.1, min(mu - (k_val * sigma), 0.9))
            # 해당 임계값 이상인 프레임 인덱스 추출
            s_idx = [i for i, c in enumerate(all_max_conf) if c >= th]
            
            video_temp_results.append({
                'Case': 'Proposed', 'Video': video_name, 'Alpha': a_val, 'Threshold': round(th, 3),
                'Size_MB': round(sum(all_buf_hq_size[i] for i in s_idx)/(1024*1024), 2),
                'Total_Frames': total_frames, 'Sent_Frames': len(s_idx),
                'Det_Frames': sum(all_has_det_hq[i] for i in s_idx), 'Baseline_Det_Frames': baseline_det_frames
            })

        # 비율 계산 및 CSV 저장
        for r in video_temp_results:
            r['Sent_Ratio_Pct'] = round((r['Sent_Frames'] / total_frames) * 100, 2) if total_frames > 0 else 0
            r['Det_Recall_Pct'] = round((r['Det_Frames'] / baseline_det_frames) * 100, 2) if baseline_det_frames > 0 else 0

        save_to_csv(video_temp_results)
        gc.collect() 
        print(f"    ✓ {video_name} 완료 → CSV 데이터 기록 및 이미지 {saved_count}장 확보")

    print(f"\n{'='*60}\n전체 실험 종료!\n결과 파일: {CSV_PATH}\n{'='*60}")

if __name__ == "__main__":
    run_full_experiment()