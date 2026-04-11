import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# =========================================================
# [설정]
# =========================================================
FILE_PATH = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\hyewon\agent\result\0410\experiment_results_0410.csv"
OUTPUT_DIR = r"C:\Users\이혜원\OneDrive\바탕 화면\졸논\EdgeAI-Realtime-Analytics\hyewon\agent\result\0410"

sns.set_theme(style="whitegrid", font_scale=1.1)

order = ['Baseline', 'Uniform(Q20)', 'Proposed (α=0.01)',
         'Proposed (α=0.05)', 'Proposed (α=0.1)', 'Proposed (α=0.2)']
palette = {
    'Baseline': '#2c2c2c',
    'Uniform(Q20)': '#aaaaaa',
    'Proposed (α=0.01)': '#1f77b4',
    'Proposed (α=0.05)': '#ff7f0e',
    'Proposed (α=0.1)': '#2ca02c',
    'Proposed (α=0.2)': '#d62728'
}

# =========================================================
# [함수 1] 데이터 로드
# =========================================================
def load_and_preprocess():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)
    df['Case_Label'] = df.apply(
        lambda x: f"Proposed (α={x['Alpha']})" if x['Case'] == 'Proposed' else x['Case'], axis=1
    )
    return df

# =========================================================
# [함수 2] 라인 그래프 (그래프 1~3)
# =========================================================
def save_individual_plot(df, y_col, title, filename, is_pct=False):
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=df, x='Video', y=y_col,
                      hue='Case_Label', hue_order=order,
                      palette=palette, marker='o', markersize=9, linewidth=2.5)
    plt.title(title, fontsize=15, fontweight='bold', pad=20)

    y_label_map = {'Size_MB': 'Size (MB)', 'Sent_Ratio_Pct': 'Sent Ratio (%)', 'Det_Recall_Pct': 'Det Recall (%)'}
    plt.ylabel(y_label_map.get(y_col, y_col), fontsize=12, fontweight='bold')
    plt.xlabel('Video File (VIRAT)', fontsize=12, fontweight='bold')
    plt.tick_params(axis='x', rotation=45)

    if is_pct:
        plt.ylim(-5, 110)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    plt.legend(title='Experiment Case', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10, title_fontsize=11)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" ✓ 저장 완료: {save_path}")

# =========================================================
# [함수 3] 트레이드오프 산점도 (그래프 4)
# =========================================================
def save_tradeoff_plot(df):
    # Proposed 케이스만 필터링
    proposed_df = df[df['Case'] == 'Proposed'].copy()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=proposed_df, x='Sent_Ratio_Pct', y='Det_Recall_Pct',
                         hue='Case_Label', hue_order=[o for o in order if 'Proposed' in o],
                         palette={k: v for k, v in palette.items() if 'Proposed' in k},
                         s=120, zorder=5)

    # 대각선 기준선 (전송률 = 탐지유지율)
    plt.plot([0, 100], [0, 100], 'k--', linewidth=1.2, alpha=0.4, label='y = x')

    plt.title('④ Tradeoff: Sent Ratio vs Det Recall (%)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Sent Ratio (%)', fontsize=11, fontweight='bold')
    plt.ylabel('Det Recall (%)', fontsize=11, fontweight='bold')
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    plt.legend(title='Alpha', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    save_path = os.path.join(OUTPUT_DIR, '04_tradeoff.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" ✓ 저장 완료: {save_path}")

# =========================================================
# [메인]
# =========================================================
if __name__ == "__main__":
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df = load_and_preprocess()
        print(f"총 {len(df)}개 데이터 로드 완료.\n")

        save_individual_plot(df, 'Size_MB',
                             '① Transmission Data Size (MB)',
                             '01_transmission_size.png')

        save_individual_plot(df, 'Sent_Ratio_Pct',
                             '② Frame Sent Ratio vs Total Frames (%)',
                             '02_frame_sent_ratio.png', is_pct=True)

        save_individual_plot(df, 'Det_Recall_Pct',
                             '③ Object Detection Recall vs Baseline (%)',
                             '03_detection_recall.png', is_pct=True)

        save_tradeoff_plot(df)  # ← 별도 함수로 분리

        print(f"\n{'='*50}")
        print(f"모든 그래프가 저장되었습니다: {OUTPUT_DIR}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n[Error] 오류: {e}")