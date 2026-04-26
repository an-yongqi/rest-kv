# %%
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-white')
# Data for the plot
context_lengths = ['4k', '8k', '16k', '32k', '64k', '128k', '256k']
SnapKV = [16.19, 16.64, 17.55, 19.36, 22.99, 30.24, 44.75]
ReST_KV = [16.31, 16.76, 17.71, 19.58, 23.21, 30.56, 45.17]
FullCache_FlashAttn2 = [16.86, 17.86, 19.85, 23.83, 31.80, 47.73, 79.59]

colors = ['#2a9d8f', '#e76f51', '#B1B2FF']
special_color = '#EAEAEA'  # Special color for the last data point
special_label = 'Full Cache OOM (N/A)'

# Bar width and positions
bar_width = 0.25
index = np.arange(len(context_lengths))

# Plotting
fig, ax = plt.subplots(figsize=(9, 5))

# Plotting each bar
ax.bar(index - bar_width, SnapKV, bar_width, label='SnapKV', color=colors[0])
ax.bar(index, ReST_KV, bar_width, label='ReST-KV', color=colors[1])
ax.bar(index[:-1] + bar_width, FullCache_FlashAttn2[:-1], bar_width, label='Full Cache w. FlashAttn-2', color=colors[2])

# Plot the last data point with a different color and label
ax.bar(index[-1] + bar_width, FullCache_FlashAttn2[-1], bar_width, label=special_label, color=special_color, hatch='//')

# Labeling the axes
ax.set_xlabel('Context Length', fontsize=22)
ax.set_ylabel('Peak Memory (GB)', fontsize=22)

# Set x-ticks
ax.set_xticks(index)
ax.set_xticklabels(context_lengths, fontsize=18)

# Set y-ticks
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
ax.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80], fontsize=16)

# Enable major ticks and minor ticks
ax.tick_params(axis='x', which='major', length=4, width=1, direction='out', grid_color='r', grid_alpha=0.5)
ax.tick_params(axis='y', which='major', length=4, width=1, direction='out', grid_color='r', grid_alpha=0.5)

# Add legend
plt.legend(loc="upper left", fontsize=18, frameon=True, ncol=1, bbox_to_anchor=(0.0, 1.0))

# Displaying the plot
plt.tight_layout()
plt.savefig("figures/memory_mistral.pdf", dpi=300)
plt.show()

# Data for the plot
context_lengths = ['4k', '8k', '16k', '32k', '64k', '128k', '256k']
SnapKV = [0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032]
ReST_KV = [0.031, 0.030, 0.032, 0.033, 0.032, 0.034, 0.034]
FullCache_FlashAttn2 = [0.032, 0.041, 0.063, 0.101, 0.184, 0.342, 0.702]

colors = ['#2a9d8f', '#e76f51', '#B1B2FF']
special_color = '#EAEAEA'  # Special color for the last data point
special_label = 'Full Cache OOM (N/A)'

# Bar width and positions
bar_width = 0.25
index = np.arange(len(context_lengths))

# Plotting
fig, ax = plt.subplots(figsize=(9, 5))

# Plotting each bar
ax.bar(index - bar_width, SnapKV, bar_width, label='SnapKV', color=colors[0])
ax.bar(index, ReST_KV, bar_width, label='ReST-KV', color=colors[1])
ax.bar(index[:-1] + bar_width, FullCache_FlashAttn2[:-1], bar_width, label='Full Cache w. FlashAttn-2', color=colors[2])

# Plot the last data point with a different color and label
ax.bar(index[-1] + bar_width, FullCache_FlashAttn2[-1], bar_width, label=special_label, color=special_color, hatch='//')

# Labeling the axes
ax.set_xlabel('Context Length', fontsize=22)
ax.set_ylabel('Decoding Latency (s)', fontsize=22)

# Set x-ticks
ax.set_xticks(index)
ax.set_xticklabels(context_lengths, fontsize=18)

# Set y-ticks
ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15", "0.20", "0.20", "0.30", "0.35"], fontsize=16)

ax.set_ylim(0, 0.35)

# Enable major ticks and minor ticks
ax.tick_params(axis='x', which='major', length=4, width=1, direction='out', grid_color='r', grid_alpha=0.5)
ax.tick_params(axis='y', which='major', length=4, width=1, direction='out', grid_color='r', grid_alpha=0.5)

# Add legend
plt.legend(loc="upper left", fontsize=18, frameon=True, ncol=1, bbox_to_anchor=(0.0, 1.0))

# Displaying the plot
plt.tight_layout()
plt.savefig("figures/latency_mistral.pdf", dpi=300)
plt.show()

# %%
