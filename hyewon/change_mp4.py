# 이미지 시퀀스를 mp4로 변환하는 스크립트
import cv2, glob, os

seq_root = r"경로\DETRAC-Images"
for seq_name in os.listdir(seq_root):
    frames = sorted(glob.glob(os.path.join(seq_root, seq_name, "*.jpg")))
    if not frames: continue
    
    h, w = cv2.imread(frames[0]).shape[:2]
    out = cv2.VideoWriter(
        f"{seq_name}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h)
    )
    for f in frames:
        out.write(cv2.imread(f))
    out.release()
    print(f"{seq_name} 변환 완료")