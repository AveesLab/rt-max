import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Argument parser setup
parser = argparse.ArgumentParser(description='Visualize system operation per CPU core.')
parser.add_argument('-reverse', type=int, help='is Reverse?')
parser.add_argument('-glayer', type=int, required=True, help='GPU layer value')
parser.add_argument('-rlayer', type=int, help='Reclaiming layer value')
args = parser.parse_args()

# Determine the CSV file path based on the input arguments
if args.rlayer is not None and args.glayer is not None:
    file_path = f"./measure/cpu-reclaiming/densenet201/{str(args.glayer).zfill(3)}glayer/cpu-reclaiming_{str(args.rlayer).zfill(3)}rlayer.csv"
elif args.glayer is not None and args.rlayer is None:
    if args.reverse is None:
        file_path = f"./measure/gpu-accel_gpu/densenet201/gpu-accel_{str(args.glayer).zfill(3)}glayer.csv"
    else:
        file_path = f"./measure/gpu-accel-reverse/densenet201/gpu-accel-reverse_{str(args.glayer).zfill(3)}glayer.csv"
else:
    raise ValueError("The 'glayer' argument must be provided.")


# Load the CSV file
df = pd.read_csv(file_path)

# Define a function to draw the process boxes
def draw_process(ax, y, start, end, label, color):
    width = end - start
    ax.barh(y, width, left=start, height=0.4, color=color, label=label, align='center')

# Setup the plot
fig, ax = plt.subplots(figsize=(50, 10))
colors = {'Preprocess': 'skyblue', 'GPU Inference': 'limegreen', 'Reclaim Inference': 'violet', 'CPU Inference': 'orange', 'Postprocess': 'salmon'}
core_ids = df['core_id'].unique()
core_ids.sort()
core_ids = core_ids[::-1]  # Reverse the array after sorting
y_ticks = np.arange(len(core_ids))

# Draw each process for each core
for i, core_id in enumerate(core_ids):
    for j in range(9, 15):
        core_data = df[df['core_id'] == core_id].iloc[j]
        draw_process(ax, i, core_data['start_preprocess'], core_data['end_preprocess'], 'Preprocess', colors['Preprocess'])
        draw_process(ax, i, core_data['start_gpu_infer'], core_data['end_gpu_infer'], 'GPU Inference', colors['GPU Inference'])        
        draw_process(ax, i, core_data['start_cpu_infer'], core_data['end_cpu_infer'], 'CPU Inference', colors['CPU Inference'])        
        if args.rlayer is not None:  # Draw this process only if rlayer is provided
            draw_process(ax, i, core_data['start_reclaim_infer'], core_data['end_reclaim_infer'], 'Reclaim Inference', colors['Reclaim Inference'])
        draw_process(ax, i, core_data['start_postprocess'], core_data['end_postprocess'], 'Postprocess', colors['Postprocess'])

# Formatting the plot
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"Core {id}" for id in core_ids])
ax.set_xlabel("Time")
ax.set_title("System Operation Visualization per CPU Core")

# Adjust legend to avoid duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicates
ax.legend(by_label.values(), by_label.keys())
plt.tight_layout()


# Create the directory if it doesn't exist
directory = "graph"
if not os.path.exists(directory):
    os.makedirs(directory)

# Generate filename with current datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{directory}/graph_{current_time}.png"


# Save the figure
plt.savefig(filename)
plt.show()

plt.close(fig)  # Close the figure after saving to free up resources
