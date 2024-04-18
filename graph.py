import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the CSV file to see its content and structure
# df = pd.read_csv('gpu-accel-mp_200glayer.csv')
# df = pd.read_csv('gpu-accel-mp_200glayer_sleep.csv')
df = pd.read_csv('measure/cpu-reclaiming-mp/densenet201/200glayer/cpu-reclaiming-mp_250rlayer.csv')
# Define a function to draw the process boxes
def draw_process(ax, y, start, end, label, color):
    width = end - start
    ax.barh(y, width, left=start, height=0.4, color=color, label=label, align='center')
# Setup the plot
fig, ax = plt.subplots(figsize=(20, 5))
colors = {'Preprocess': 'skyblue', 'GPU Inference': 'limegreen','Reclaim Inference': 'violet',  'CPU Inference': 'orange', 'Postprocess': 'salmon'}
core_ids = df['core_id'].unique()
core_ids.sort()
y_ticks = np.arange(len(core_ids))
print(len(df['core_id']))
# Draw each process for each core
for i, core_id in enumerate(core_ids):
    for j in range(3, 8):
        core_data = df[df['core_id'] == core_id].iloc[j]
        draw_process(ax, i, core_data['start_preprocess'], core_data['end_preprocess'], 'Preprocess', colors['Preprocess'])
        #print(core_data['start_preprocess'])
        draw_process(ax, i, core_data['start_gpu_infer'], core_data['end_gpu_infer'], 'GPU Inference', colors['GPU Inference'])
        draw_process(ax, i, core_data['start_reclaim_infer'], core_data['end_reclaim_infer'], 'Reclaim Inference', colors['Reclaim Inference'])
        draw_process(ax, i, core_data['start_cpu_infer'], core_data['end_cpu_infer'], 'CPU Inference', colors['CPU Inference'])
        draw_process(ax, i, core_data['start_postprocess'], core_data['end_postprocess'], 'Postprocess', colors['Postprocess'])
# Formatting the plot
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"Core {id}" for id in core_ids])
ax.set_xlabel("Time")
ax.set_title("System Operation Visualization per CPU Core")
# 범례는 중복 표시를 방지하기 위해 조정 필요
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # 중복 제거
#plt.xlim(37314835, 37316763)
plt.tight_layout()

plt.show()
