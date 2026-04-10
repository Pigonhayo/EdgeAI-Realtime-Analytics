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
RESULT_DIR = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\hyewon\agent\result"
CSV_PATH = os.path.join(RESULT_DIR, "experiment_results_0409.csv")
MODEL = YOLO('yolov8n.pt')
CALIB_FRAMES = 200

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

    print(f"총 {len(video_files)}개 비디오 분석 시작")
    print(f"저장 위치: {CSV_PATH}\n")

    for v_idx, v_path in enumerate(video_files):
        video_name = os.path.basename(v_path)
        total_frames_est = get_total_frames(v_path)
        if total_frames_est == 0:
            print(f"Skipping {video_name}: 열 수 없음")
            continue

        print(f"\n[{v_idx+1}/{len(video_files)}] {video_name} ({total_frames_est}프레임)")

        # -------------------------------------------------------
        # [핵심] 영상 1회 순회: YOLO 결과 전체 저장
        # -------------------------------------------------------
        cap = cv2.VideoCapture(v_path)
        
        # 프레임별 결과를 저장할 리스트
        all_max_conf = []   # 프레임별 max confidence
        all_has_det = []    # 프레임별 탐지 여부
        all_buf_hq = []     # 프레임별 고화질 인코딩 크기 (bytes)
        all_buf_lq = []     # 프레임별 저화질 인코딩 크기 (bytes)

        with tqdm(total=total_frames_est, desc="  [1/2] YOLO 분석", unit="f", ncols=80) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # YOLO 추론 (1회만)
                res = MODEL(frame, verbose=False)
                has_det = len(res[0].boxes) > 0
                max_c = res[0].boxes.conf.max().item() if has_det else 0.0

                # 인코딩 크기 계산
                _, buf_hq = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                _, buf_lq = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

                all_max_conf.append(max_c)
                all_has_det.append(has_det)
                all_buf_hq.append(len(buf_hq))
                all_buf_lq.append(len(buf_lq))

                pbar.update(1)
        cap.release()

        total_frames = len(all_max_conf)
        print(f"     → 총 {total_frames}프레임 분석 완료")

        # -------------------------------------------------------
        # Calibration: 초반 200프레임의 confidence 분포 계산
        # -------------------------------------------------------
        calib_scores = [all_max_conf[i] for i in range(min(CALIB_FRAMES, total_frames))
                        if all_has_det[i]]
        mu, sigma = (np.mean(calib_scores), np.std(calib_scores)) if calib_scores else (0.5, 0.1)
        print(f"     → Calibration: μ={mu:.3f}, σ={sigma:.3f}")

        # -------------------------------------------------------
        # [2/2] 각 케이스 결과 계산 (YOLO 재실행 없음)
        # -------------------------------------------------------
        video_temp_results = []

        with tqdm(total=5, desc="  [2/2] 케이스 계산", unit="case", ncols=80) as pbar:

            # Case 1: Baseline
            baseline_det_frames = sum(all_has_det)
            size_baseline_mb = round(sum(all_buf_hq) / (1024*1024), 2)
            video_temp_results.append({
                'Case': 'Baseline', 'Video': video_name, 'Alpha': '-', 'Threshold': '-',
                'Size_MB': size_baseline_mb, 'Total_Frames': total_frames,
                'Sent_Frames': total_frames, 'Det_Frames': baseline_det_frames,
                'Baseline_Det_Frames': baseline_det_frames
            })
            pbar.update(1)

            # Case 2: Uniform (Q20)
            det_uniform = sum(all_has_det)  # 모든 프레임 전송이므로 동일
            size_uniform_mb = round(sum(all_buf_lq) / (1024*1024), 2)
            video_temp_results.append({
                'Case': 'Uniform(Q20)', 'Video': video_name, 'Alpha': '-', 'Threshold': '-',
                'Size_MB': size_uniform_mb, 'Total_Frames': total_frames,
                'Sent_Frames': total_frames, 'Det_Frames': det_uniform,
                'Baseline_Det_Frames': baseline_det_frames
            })
            pbar.update(1)

            # Case 3: Proposed (Multi-Alpha)
            alphas = {0.01: 2.33, 0.05: 1.645, 0.1: 1.28, 0.2: 0.84}
            for alpha, k in alphas.items():
                threshold = max(0.1, min(mu - (k * sigma), 0.9))

                sent_frames = [i for i, c in enumerate(all_max_conf) if c >= threshold]
                size_proposed_mb = round(sum(all_buf_hq[i] for i in sent_frames) / (1024*1024), 2)
                det_proposed = sum(all_has_det[i] for i in sent_frames)

                video_temp_results.append({
                    'Case': 'Proposed', 'Video': video_name, 'Alpha': alpha,
                    'Threshold': round(threshold, 3),
                    'Size_MB': size_proposed_mb, 'Total_Frames': total_frames,
                    'Sent_Frames': len(sent_frames), 'Det_Frames': det_proposed,
                    'Baseline_Det_Frames': baseline_det_frames
                })
                pbar.update(1)

        # 비율 계산
        for r in video_temp_results:
            r['Sent_Ratio_Pct'] = round((r['Sent_Frames'] / total_frames) * 100, 2) if total_frames > 0 else 0
            r['Det_Recall_Pct'] = round((r['Det_Frames'] / baseline_det_frames) * 100, 2) if baseline_det_frames > 0 else 0

        save_to_csv(video_temp_results)
        print(f"  ✓ {video_name} 완료 → CSV 저장")

    print(f"\n{'='*60}")
    print(f"전체 실험 종료! 결과: {CSV_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_full_experiment()