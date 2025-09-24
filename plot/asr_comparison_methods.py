import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D





methods = [
    "GCG", "PRP", "COLD","DRL",
    "PAP", "AdvPrefix", "I-GCG", "Ours"
]


colors = [
    "#f78dd7",  
    "#e94f4f",  
    "#f38d34", 
    "#f76345",  
    "#34a4f5", 
    "#2cdff3", 
    "#a150ed",  
    "#63e763",  
]

hatches = [
    '', '////', '\\\\', 'xxxx', '...', '**', 'oo', '++'
]

datasets = ['advbench', 'harmBench']
x = [0.4,0.6]  # [0, 1]

perplexity_dpr = np.array([
    [13,11],    # TemplateJailbreak
    [6, 6],    # SelfCipher
    [26, 17],       # GCG
    [25, 27],       # TemplatePrompt
    [1, 7],    # DeepInception
    [17, 16],    # Jailbroken
    [4, 7],    # QEPrompt
    [35, 44],    # MixAsking
])

lmguard_dpr = np.array([
    [27, 52],
    [18, 23],
    [28, 18],
    [16, 14],
    [72, 52],
    [36, 47],
    [35, 86],
    [84, 89],
])

bar_width = 0.012
group_width = bar_width * len(methods)
offsets = np.linspace(-group_width/2, group_width/2, len(methods))

fig=plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
axes = [ax1, ax2]

for ax_idx, (ax, data, title) in enumerate(zip(
    axes,
    [perplexity_dpr, lmguard_dpr],
    ['GPTFuzz  ASR(%)', '...  ASR(%)']
)):
    for i, (method, color, hatch) in enumerate(zip(methods, colors, hatches)):
        for j in range(len(datasets)):
            try:
                height = data[i][j]
                ax.bar(x[j] + offsets[i], height, width=bar_width,
                       color=color, hatch=hatch, edgecolor='black',
                       label=method if ax_idx == 0 and j == 0 else "")
            except IndexError:
                continue
    ax.set_ylabel(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

model_legend = [Line2D([0], [0], color='white', label='Llama-3-70B')]

axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets,fontsize=12)
ax1.text(0.5, -0.05, '(a)', transform=ax1.transAxes,fontsize=12, va='top', ha='left')
ax1.legend(
    handles=model_legend,
    loc='upper left',
    frameon=True,
    fontsize=12,
    handlelength=0,  
    handletextpad=0  
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(datasets,fontsize=12)
axes[1].set_xlabel("")
ax2.text(0.5, -0.05, '(b)', transform=ax2.transAxes,fontsize=12, va='top', ha='left')
ax1.set_xlim(0.3, 0.7)  
ax2.set_xlim(0.3, 0.7)  
ax1.set_ylim(0, 50)
ax2.set_ylim(0, 100)

legend_handles = []
for method, color, hatch in zip(methods, colors, hatches):
    patch = mpatches.Patch(facecolor=color, edgecolor='black',
                           hatch=hatch, label=method, linewidth=0.5)
    legend_handles.append(patch)

fig.legend(handles=legend_handles,
           loc='upper left',
           ncol=4,
           fontsize=12,
           bbox_to_anchor=(0.1, 0.90,0.8,0.1),
           bbox_transform=fig.transFigure,
           mode='expand',
           columnspacing=1.5,
           handlelength=2,
           handleheight=1.0,
           frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.95])  

plt.savefig("asr_comparison_methods_53.pdf")
plt.show()
