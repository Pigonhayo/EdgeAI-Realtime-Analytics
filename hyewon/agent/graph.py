import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# [설정 1] 경로 앞에 r을 붙여서 윈도우 경로 인식을 보장합니다.
FILE_PATH = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\result\experiment_results_0409.csv"
OUTPUT_DIR = "./result/plots" # 결과 그래프 저장 폴더

# [설정 2] 공통 스타일 및 색상 팔레트 정의
sns.set_theme(style="whitegrid", font_scale=1.1) # 전체 폰트 크기 살짝 키움 (가독성)

order = ['Baseline', 'Uniform(Q20)', 'Proposed (α=0.01)',
         'Proposed (α=0.05)', 'Proposed (α=0.1)', 'Proposed (α=0.2)']
palette = {
    'Baseline': '#2c2c2c',       # 검정
    'Uniform(Q20)': '#aaaaaa',  # 회색
    'Proposed (α=0.01)': '#1f77b4', # 파랑
    'Proposed (α=0.05)': '#ff7f0e', # 주황
    'Proposed (α=0.1)': '#2ca02c',  # 초록
    'Proposed (α=0.2)': '#d62728'   # 빨강
}

def load_and_preprocess():
    """데이터 로드 및 전처리"""
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {FILE_PATH}")
    
    df = pd.read_csv(FILE_PATH)
    # Case와 Alpha를 조합해 라벨 생성
    df['Case_Label'] = df.apply(
        lambda x: f"Proposed (α={x['Alpha']})" if x['Case'] == 'Proposed' else x['Case'], axis=1
    )
    return df

def save_individual_plot(df, y_col, title, filename, is_pct=False):
    """개별 그래프 생성 및 저장 함수"""
    # 1. 그래프 크기 설정 (개별 저장용으로 최적화)
    plt.figure(figsize=(12, 6)) 
    
    # 2. Seaborn lineplot 그리기
    # 학술용 그래프는 마커 크기(markersize)와 선 굵기(linewidth)를 조금 키우는 것이 좋습니다.
    ax = sns.lineplot(data=df, x='Video', y=y_col,
                       hue='Case_Label', hue_order=order,
                       palette=palette, marker='o', markersize=9, linewidth=2.5)
    
    # 3. 그래프 꾸미기 (제목, 축 라벨 등)
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    
    # y축 라벨 설정
    y_label = y_col
    if y_col == 'Size_MB': y_label = 'Size (MB)'
    elif y_col == 'Sent_Ratio_Pct': y_label = 'Sent Ratio (%)'
    elif y_col == 'Det_Recall_Pct': y_label = 'Det Recall (%)'
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    
    # x축 설정
    plt.xlabel('Video File (VIRAT)', fontsize=12, fontweight='bold')
    plt.tick_params(axis='x', rotation=45) # 비디오 이름이 겹치지 않게 회전
    
    # 퍼센트 단위 설정 (% 기호 추가)
    if is_pct:
        plt.ylim(-5, 110) # 100% 위쪽 여유 공간
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    
    # 4. 범례(Legend) 설정: 그래프 바깥 오른쪽 상단에 배치
    plt.legend(title='Experiment Case', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title_fontsize=11)
    
    # 5. 저장 및 닫기 (tight_layout은 범례가 잘리지 않게 중요함)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight') # 고해상도(200dpi) 저장
    plt.close() # 메모리 해제
    print(f" ✓ 그래프 저장 완료: {save_path}")

# =========================================================
# 메인 실행부
# =========================================================
if __name__ == "__main__":
    try:
        # 데이터 폴더 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 데이터 로드
        experiment_df = load_and_preprocess()
        print(f"총 {len(experiment_df)}개의 실험 데이터 로드 완료.\n")
        
        # --- 그래프 1: 전송 데이터량 (Size_MB) ---
        save_individual_plot(experiment_df, 'Size_MB', 
                             '① Transmission Data Size (MB)', 
                             '01_transmission_size.png')
        
        # --- 그래프 2: 프레임 전송률 (Sent_Ratio_Pct) ---
        save_individual_plot(experiment_df, 'Sent_Ratio_Pct', 
                             '② Frame Sent Ratio vs Total Frames (%)', 
                             '02_frame_sent_ratio.png', is_pct=True)
        
        # --- 그래프 3: 탐지 정확도 유지율 (Det_Recall_Pct) ← 핵심 ---
        save_individual_plot(experiment_df, 'Det_Recall_Pct', 
                             '③ Object Detection Recall vs Baseline (%)', 
                             '03_detection_recall.png', is_pct=True)
        
        print(f"\n{'='*50}")
        print(f"모든 개별 그래프가 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n[Error] 오류가 발생했습니다: {e}")