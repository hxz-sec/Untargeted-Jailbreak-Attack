import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


iters = ['0','25', '50', '75', '100']
gptfuzz_data = {
    'Llama3': [0, 71, 83, 90, 89],
    'Vicuna': [0, 63, 82, 79, 88],
    'Mistral': [0, 86, 89, 93, 88],
    'Qwen2.5': [0, 36, 42, 55, 74]
}

harmbench_data = {
    'Llama3': [0, 57, 64, 59, 57],
    'Vicuna': [0, 47, 66, 59, 59],
    'Mistral': [0, 74, 83, 87, 85],
    'Qwen2.5': [0, 30, 36, 40, 55]
}


def plot_attack(data, ylabel_text):
    models = list(data.keys())
    x = np.arange(len(iters))
    colors = sns.color_palette("Set2", len(models))
    markers = ['o', 's', 'D', '^'] 

    for i, model in enumerate(models):
        plt.plot(x, data[model],
                 label=model,
                 color=colors[i],
                 marker=markers[i % len(markers)],
                 linewidth=2,
                 markersize=5)

    plt.xticks(x, iters, fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel(ylabel_text, fontsize=8)
    plt.ylim(0, max(max(v) for v in data.values()) * 1.15)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=8, frameon=True,loc='lower right')


plt.figure(figsize=(5, 3))

plt.subplot(1, 2, 1)
plot_attack(gptfuzz_data, 'ASR-G (%)')
plt.text(0.43, -0.2, 'iters', transform=plt.gca().transAxes, fontsize=8)
plt.text(0.46, -0.3, '(a)', transform=plt.gca().transAxes, fontsize=8)

plt.subplot(1, 2, 2)
plot_attack(harmbench_data, 'ASR-H (%)')
plt.text(0.43, -0.2, 'iters', transform=plt.gca().transAxes, fontsize=8)
plt.text(0.46, -0.3, '(b)', transform=plt.gca().transAxes, fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 1])  
plt.savefig("attack_asr_comparison_16.pdf", dpi=300)
