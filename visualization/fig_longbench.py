# %%
import matplotlib.pyplot as plt

# Sample data (replace with actual values from your dataset)
cache_sizes = [64, 128, 256, 512, 1024]
streaming_llm = [29.61, 31.14, 32.11, 33.84, 35.16]
h2o = [29.52, 30.56, 30.96, 34.53, 36.27]
tova = [30.63, 32.54, 35.07, 35.59, 37.55]
snap_kv = [31.48, 33.98, 36.07, 37.47, 38.61]
rest_kv = [33.56, 35.86, 36.93, 38.33, 38.94]

# Plotting
plt.figure(figsize=(8, 6))

colors = ['#F4CE14', '#00b4d8', '#0582ca', '#2a9d8f', '#e76f51']

# Plot each line with its specific style
plt.plot(cache_sizes, streaming_llm, label="StreamingLLM", linestyle="-", marker="o", color=colors[0])
plt.plot(cache_sizes, h2o, label="H2O", linestyle="-", marker="o", color=colors[1])
plt.plot(cache_sizes, tova, label="TOVA", linestyle="-", marker="o", color=colors[2])
plt.plot(cache_sizes, snap_kv, label="SnapKV", linestyle="-", marker="o", color=colors[3])
plt.plot(cache_sizes, rest_kv, label="ReST-KV", linestyle="-", marker="^", color=colors[4], linewidth=3, markersize=8)

# Add dashed horizontal line for "Full Cache"
plt.axhline(y=39.96, color='gray', linestyle='--', label='Full Cache')

# Add labels, title, and grid
plt.xlabel("Cache Size (L)", fontsize=18)
plt.ylabel("Average Score", fontsize=18)
plt.title("Llama3.1-8B-Instruct", fontsize=24)
plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

# Add legend
plt.legend(loc="lower right", fontsize=14, frameon=True, ncol=2, bbox_to_anchor=(1, 0.0))

# Set x-axis to log scale if it matches your figure style
plt.xscale("log", base=10)

# Customize tick marks
x_ticks = [64, 128, 256, 512, 1024]
plt.gca().set_xticks(x_ticks)  # Set the tick positions on log scale
plt.gca().set_xticklabels([str(i) for i in x_ticks], fontsize=16, rotation=45)  # Set the labels as the specific values

y_ticks = [28, 30, 32, 34, 36, 38, 40]
plt.yticks(y_ticks, fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig("figures/longbench_llama3.1.pdf", dpi=300)
plt.show()

# Sample data (replace with actual values from your dataset)
cache_sizes = [64, 128, 256, 512, 1024]
streaming_llm = [11.83, 21.69, 27.15, 28.86, 29.92]
h2o = [12.42, 22.43, 27.95, 29.15, 30.08]
tova = [20.20, 27.41, 30.16, 30.24, 31.96]
snap_kv = [22.96, 27.41, 30.90, 32.18, 32.99]
rest_kv = [25.54, 29.99, 31.51, 32.38, 32.97]

# Plotting
plt.figure(figsize=(8, 6))

colors = ['#F4CE14', '#00b4d8', '#0582ca', '#2a9d8f', '#e76f51']

# Plot each line with its specific style
plt.plot(cache_sizes, streaming_llm, label="StreamingLLM", linestyle="-", marker="o", color=colors[0])
plt.plot(cache_sizes, h2o, label="H2O", linestyle="-", marker="o", color=colors[1])
plt.plot(cache_sizes, tova, label="TOVA", linestyle="-", marker="o", color=colors[2])
plt.plot(cache_sizes, snap_kv, label="SnapKV", linestyle="-", marker="o", color=colors[3])
plt.plot(cache_sizes, rest_kv, label="ReST-KV", linestyle="-", marker="^", color=colors[4], linewidth=3, markersize=8)

# Add dashed horizontal line for "Full Cache"
plt.axhline(y=33.37, color='gray', linestyle='--', label='Full Cache')

# Add labels, title, and grid
plt.xlabel("Cache Size (L)", fontsize=18)
plt.ylabel("Average Score", fontsize=18)
plt.title("Llama2-7B-Chat", fontsize=24)
plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

# Add legend
plt.legend(loc="lower right", fontsize=14, frameon=True, ncol=2, bbox_to_anchor=(1, 0.0))

# Set x-axis to log scale if it matches your figure style
plt.xscale("log", base=10)

# Customize tick marks
x_ticks = [64, 128, 256, 512, 1024]
plt.gca().set_xticks(x_ticks)  # Set the tick positions on log scale
plt.gca().set_xticklabels([str(i) for i in x_ticks], fontsize=16, rotation=45)  # Set the labels as the specific values

y_ticks = [10, 14, 18, 22, 26, 30, 34]
plt.yticks(y_ticks, fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig("figures/longbench_llama2.pdf", dpi=300)
plt.show()

# Sample data (replace with actual values from your dataset)
cache_sizes = [64, 128, 256, 512, 1024]
streaming_llm = [34.24, 36.03, 37.71, 39.68, 41.19]
h2o = [34.93, 36.31, 37.73, 41.42, 43.69]
tova = [36.50, 39.76, 42.28, 42.85, 46.53]
snap_kv = [37.53, 41.38, 43.79, 45.66, 46.83]
rest_kv = [40.05, 43.52, 45.27, 46.47, 47.18]

# Plotting
plt.figure(figsize=(8, 6))

colors = ['#F4CE14', '#00b4d8', '#0582ca', '#2a9d8f', '#e76f51']

# Plot each line with its specific style
plt.plot(cache_sizes, streaming_llm, label="StreamingLLM", linestyle="-", marker="o", color=colors[0])
plt.plot(cache_sizes, h2o, label="H2O", linestyle="-", marker="o", color=colors[1])
plt.plot(cache_sizes, tova, label="TOVA", linestyle="-", marker="o", color=colors[2])
plt.plot(cache_sizes, snap_kv, label="SnapKV", linestyle="-", marker="o", color=colors[3])
plt.plot(cache_sizes, rest_kv, label="ReST-KV", linestyle="-", marker="^", color=colors[4], linewidth=3, markersize=8)

# Add dashed horizontal line for "Full Cache"
plt.axhline(y=48.11, color='gray', linestyle='--', label='Full Cache')

# Add labels, title, and grid
plt.xlabel("Cache Size (L)", fontsize=18)
plt.ylabel("Average Score", fontsize=18)
plt.title("Mistral-7B-Instruct-v0.3", fontsize=24)
plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

# Add legend
plt.legend(loc="lower right", fontsize=14, frameon=True, ncol=2, bbox_to_anchor=(1, 0.0))

# Set x-axis to log scale if it matches your figure style
plt.xscale("log", base=10)

# Customize tick marks
x_ticks = [64, 128, 256, 512, 1024]
plt.gca().set_xticks(x_ticks)  # Set the tick positions on log scale
plt.gca().set_xticklabels([str(i) for i in x_ticks], fontsize=16, rotation=45)  # Set the labels as the specific values

y_ticks = [32, 35, 38, 41, 44, 47, 49]
plt.yticks(y_ticks, fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig("figures/longbench_mistral.pdf", dpi=300)
plt.show()

# %%
