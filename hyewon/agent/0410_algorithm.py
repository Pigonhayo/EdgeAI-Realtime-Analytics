import cv2
import numpy as np
import os
import pandas as pd
import glob
from ultralytics import YOLO
from tqdm import tqdm

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
        print(f"[Error] 비디오 없음: {FOLDER_PATH}")
        return

    os.makedirs(FRAMES_SAVE_DIR, exist_ok=True)
    print(f"총 {len(video_files)}개 비디오 분석 시작\n")

    for v_idx, v_path in enumerate(video_files):
        video_name = os.path.basename(v_path)
        total_frames_est = get_total_frames(v_path)
        if total_frames_est == 0:
            print(f"Skipping {video_name}")
            continue

        print(f"\n[{v_idx+1}/{len(video_files)}] {video_name} ({total_frames_est}프레임)")

        # -------------------------------------------------------
        # [1회차] YOLO 분석 → 크기만 저장 (메모리 절약)
        # -------------------------------------------------------
        cap = cv2.VideoCapture(v_path)
        all_max_conf = []
        all_has_det = []
        all_buf_hq_size = []  # 크기(int)만 저장
        all_buf_lq_size = []  # 크기(int)만 저장

        with tqdm(total=total_frames_est, desc="  [1/3] YOLO 분석", unit="f", ncols=80) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                res = MODEL(frame, verbose=False)
                has_det = len(res[0].boxes) > 0
                max_c = res[0].boxes.conf.max().item() if has_det else 0.0

                _, buf_hq = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                _, buf_lq = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

                all_max_conf.append(max_c)
                all_has_det.append(has_det)
                all_buf_hq_size.append(len(buf_hq))  # 크기만
                all_buf_lq_size.append(len(buf_lq))  # 크기만
                pbar.update(1)
        cap.release()

        total_frames = len(all_max_conf)

        # -------------------------------------------------------
        # Calibration → threshold 동적 도출
        # -------------------------------------------------------
        calib_scores = [all_max_conf[i] for i in range(min(CALIB_FRAMES, total_frames))
                        if all_has_det[i]]
        mu, sigma = (np.mean(calib_scores), np.std(calib_scores)) if calib_scores else (0.5, 0.1)
        print(f"     → Calibration: μ={mu:.3f}, σ={sigma:.3f}")

        threshold_final = max(0.1, min(mu - (TARGET_K * sigma), 0.9))
        sent_indices = set(i for i, c in enumerate(all_max_conf) if c >= threshold_final)
        print(f"     → threshold={threshold_final:.3f} | 선별 프레임={len(sent_indices)}/{total_frames}")

        # -------------------------------------------------------
        # [2회차] 영상 재순회 → 선별된 프레임만 이미지 저장
        # -------------------------------------------------------
        video_frame_dir = os.path.join(FRAMES_SAVE_DIR, video_name.replace('.mp4', ''))
        os.makedirs(video_frame_dir, exist_ok=True)

        cap = cv2.VideoCapture(v_path)
        frame_idx = 0
        saved_count = 0

        with tqdm(total=total_frames, desc="  [2/3] 이미지 저장", unit="f", ncols=80) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if frame_idx in sent_indices:
                    img_path = os.path.join(video_frame_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    saved_count += 1

                frame_idx += 1
                pbar.update(1)
        cap.release()
        print(f"     → {saved_count}개 프레임 이미지 저장 완료")

        # -------------------------------------------------------
        # [3단계] CSV 결과 계산
        # -------------------------------------------------------
        video_temp_results = []
        baseline_det_frames = sum(all_has_det)
        size_baseline_mb = round(sum(all_buf_hq_size) / (1024*1024), 2)

        video_temp_results.append({
            'Case': 'Baseline', 'Video': video_name, 'Alpha': '-', 'Threshold': '-',
            'Size_MB': size_baseline_mb, 'Total_Frames': total_frames,
            'Sent_Frames': total_frames, 'Det_Frames': baseline_det_frames,
            'Baseline_Det_Frames': baseline_det_frames
        })

        size_uniform_mb = round(sum(all_buf_lq_size) / (1024*1024), 2)
        video_temp_results.append({
            'Case': 'Uniform(Q20)', 'Video': video_name, 'Alpha': '-', 'Threshold': '-',
            'Size_MB': size_uniform_mb, 'Total_Frames': total_frames,
            'Sent_Frames': total_frames, 'Det_Frames': sum(all_has_det),
            'Baseline_Det_Frames': baseline_det_frames
        })

        alphas = {0.01: 2.33, 0.05: 1.645, 0.1: 1.28, 0.2: 0.84}
        for alpha, k in alphas.items():
            threshold = max(0.1, min(mu - (k * sigma), 0.9))
            sent_idx = [i for i, c in enumerate(all_max_conf) if c >= threshold]
            size_proposed_mb = round(sum(all_buf_hq_size[i] for i in sent_idx) / (1024*1024), 2)
            det_proposed = sum(all_has_det[i] for i in sent_idx)

            video_temp_results.append({
                'Case': 'Proposed', 'Video': video_name, 'Alpha': alpha,
                'Threshold': round(threshold, 3),
                'Size_MB': size_proposed_mb, 'Total_Frames': total_frames,
                'Sent_Frames': len(sent_idx), 'Det_Frames': det_proposed,
                'Baseline_Det_Frames': baseline_det_frames
            })

        for r in video_temp_results:
            r['Sent_Ratio_Pct'] = round((r['Sent_Frames'] / total_frames) * 100, 2) if total_frames > 0 else 0
            r['Det_Recall_Pct'] = round((r['Det_Frames'] / baseline_det_frames) * 100, 2) if baseline_det_frames > 0 else 0

        save_to_csv(video_temp_results)
        print(f"  ✓ {video_name} 완료 → CSV 저장")

    print(f"\n{'='*60}")
    print(f"전체 실험 종료!")
    print(f"선별 이미지 저장 위치: {FRAMES_SAVE_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_full_experiment()