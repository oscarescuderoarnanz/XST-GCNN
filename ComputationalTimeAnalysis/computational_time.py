import os
import json
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# === PATHS ===
base_path = "../XST-GNN_Architecture/step3_GCNNs/hyperparameters"
subfolders = ["correlations", "smoothness", "dtw-hgd"]

model_paths = {
    "GRU": "../Baselines/GRU/Results_GRU_optuna/summary_metrics.csv",
    "LSTM": "../Baselines/LSTM/Results_LSTM_optuna/summary_metrics.csv",
    "Transformer": "../Baselines/Transformer/Results_Transformer_optuna/summary_metrics.csv",
    "Mamba": "../Baselines/mamba/Results_Mamba_optuna/summary_metrics.csv",
    "G-GCNN": "../Baselines/G-GCNN/GRNN_bestResults"
}

label_map = {
    "E1": "GCNN-1",
    "E2": "CPG with GCNN-1",
    "E3": "STG with GCNN-1",
    "E4": "CPG with GCNN-2",
    "E5": "STG with GCNN-2"
}

if __name__ == "__main__":
    # === PLOT SETUP USING GRIDSPEC ===
    fig = plt.figure(figsize=(32, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], wspace=0.3)

    axes = [plt.subplot(gs[i]) for i in range(4)]

    # Add subplot labels (a), (b), (c), (d)
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axes):
        ax.text(
            0.5, -1.25, subplot_labels[i],
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=28
        )

    # === BASELINES SUBPLOT ===
    ax_baselines = axes[0]
    baseline_labels = []
    baseline_data = []

    for model_name, file_path in model_paths.items():
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                inference_times = df["InferenceTime"].dropna().values
            else:
                with open(file_path, "r") as f:
                    content = f.read()
                    data = ast.literal_eval(content)
                inference_times = data.get("inference_time", [])

            if len(inference_times) > 0:
                baseline_labels.append(model_name)
                baseline_data.append(inference_times)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # === GCNN METHODS SUBPLOTS ===
    for idx, folder in enumerate(subfolders):
        folder_path = os.path.join(base_path, folder)
        box_data = []
        labels = []

        for file_name in os.listdir(folder_path):
            if "#E" in file_name and file_name.endswith(".json"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)

                times = [
                    v.get("inference_time_seconds")
                    for v in data.values()
                    if isinstance(v, dict) and "inference_time_seconds" in v
                ]

                if times:
                    match = re.search(r"#E(\d+)", file_name)
                    if match:
                        e_id = f"E{match.group(1)}"
                        label = label_map.get(e_id, e_id)

                        if e_id == "E1":
                            if folder == 'dtw-hgd':
                                baseline_labels.append(f"GCNN-1 (HGD-DTW)")
                            else:
                                baseline_labels.append(f"GCNN-1 ({folder})")
                            baseline_data.append(times)
                        else:
                            labels.append(label)
                            box_data.append(times)

        ax = axes[idx + 1]
        if labels:
            sorted_pairs = sorted(
                zip(labels, box_data),
                key=lambda x: list(label_map.values()).index(x[0]) if x[0] in label_map.values() else float('inf')
            )
            labels, box_data = zip(*sorted_pairs)

        if box_data:
            ax.boxplot(box_data, labels=labels, showmeans=True)
            ax.set_title("", fontsize=28)
            ax.tick_params(axis='x', rotation=90, labelsize=28)
            ax.tick_params(axis='y', rotation=0, labelsize=28)
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylim(0, 11)
        else:
            ax.set_title(folder + "\n(No valid data)", fontsize=28)
            ax.axis('off')

    # === FINALIZE BASELINES PLOT ===
    if baseline_data:
        ax_baselines.boxplot(baseline_data, labels=baseline_labels, showmeans=True)
        ax_baselines.set_title("", fontsize=28)
        ax_baselines.tick_params(axis='x', rotation=90, labelsize=28)
        ax_baselines.set_ylabel("", fontsize=28)
        ax_baselines.tick_params(axis='y', labelsize=28)
        ax_baselines.yaxis.grid(True, linestyle='--', alpha=0.3)

    # Adjust layout
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.25, wspace=0.4)

    # Save and show plot
    plt.savefig('computational_time.pdf', bbox_inches='tight')
    plt.show()
