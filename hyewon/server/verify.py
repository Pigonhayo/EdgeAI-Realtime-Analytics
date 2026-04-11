from ultralytics import YOLO
import cv2
import glob
import os
from tqdm import tqdm
from datetime import datetime

SERVER_MODEL = YOLO('yolov8x.pt')

# 경로 설정
VIDEO_PATH = r"G:\내 드라이브\졸논\데이터셋\VIRAT\VIRAT_S_000001.mp4"
SELECTED_DIR = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\hyewon\agent\result\0410\selected_frames_alpha_01\VIRAT_S_000001"
RESULT_DIR = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\hyewon\server\result"
OUT_FILE_PATH = os.path.join(RESULT_DIR, "server_verification_results.txt")

def run_server_verification():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # -------------------------------------------------------
    # [1] Baseline: 원본 영상 전체를 v8x로 분석
    # -------------------------------------------------------
    print("[1/2] Baseline: 전체 프레임 v8x 분석 중...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    baseline_det = 0
    baseline_total = 0

    with tqdm(desc="  Baseline", unit="f", ncols=80) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            baseline_total += 1
            res = SERVER_MODEL(frame, verbose=False)
            if len(res[0].boxes) > 0:
                baseline_det += 1
            del res, frame
            pbar.update(1)
    cap.release()
    print(f"     → 전체 {baseline_total}프레임 | 탐지: {baseline_det}프레임")

    # -------------------------------------------------------
    # [2] Proposed: 선별된 프레임만 v8x로 분석
    # -------------------------------------------------------
    print("\n[2/2] Proposed: 선별 프레임 v8x 분석 중...")
    image_files = sorted(glob.glob(os.path.join(SELECTED_DIR, "*.jpg")))
    proposed_det = 0

    for img_path in tqdm(image_files, desc="  Proposed", unit="f", ncols=80):
        res = SERVER_MODEL(img_path, verbose=False)
        if len(res[0].boxes) > 0:
            proposed_det += 1
        del res

    # -------------------------------------------------------
    # [3] 결과 비교 및 파일 저장
    # -------------------------------------------------------
    recall = (proposed_det / baseline_det * 100) if baseline_det > 0 else 0
    sent_ratio = (len(image_files) / baseline_total * 100) if baseline_total > 0 else 0
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result_text = f"""
==================================================
📊 [최종 서버 검증 결과] - {now}
--------------------------------------------------
대상 영상  : {os.path.basename(VIDEO_PATH)}
Baseline  : 전체 {baseline_total}프레임 → v8x 탐지: {baseline_det}프레임
Proposed  : 선별 {len(image_files)}프레임 → v8x 탐지: {proposed_det}프레임
--------------------------------------------------
전송률     : {sent_ratio:.2f}%
정밀도유지율: {recall:.2f}%
==================================================
"""
    print(result_text)

    with open(OUT_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(result_text + "\n")

    print(f"📂 결과 저장 완료: {OUT_FILE_PATH}")

if __name__ == "__main__":
    run_server_verification()