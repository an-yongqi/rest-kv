# %%
import matplotlib.pyplot as plt

# Sample data (replace with actual values from your dataset)
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rest_kv = [34.83, 35.25, 35.69, 35.58, 35.51, 35.57, 35.39, 35.24, 35.17]

# Plotting
plt.figure(figsize=(8, 6))

colors = ['#e76f51']

# Plot each line with its specific style
plt.plot(alpha, rest_kv, label="EMA", linestyle="-", marker="^", color=colors[0], linewidth=3, markersize=8)

# Add dashed horizontal line for "Full Cache"
plt.axhline(y=34.15, color='gray', linestyle='--', label='Mean')

# Add labels, title, and grid
plt.xlabel(r"Smoothing Factor $\alpha$", fontsize=18)
plt.ylabel("Average Score", fontsize=18)
plt.title("Llama3.1-8B-Instruct", fontsize=24)
plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

plt.xticks(alpha, fontsize=16)

y_ticks = [33, 33.5, 34, 34.5, 35, 35.5, 36]
plt.yticks(y_ticks, fontsize=16)

# Add legend
plt.legend(loc="lower right", fontsize=18, frameon=True, ncol=2, bbox_to_anchor=(1, 0.0))

# Show the plot
plt.tight_layout()
plt.savefig("figures/sensitivity_alpha.pdf", dpi=300)
plt.show()

# Sample data (replace with actual values from your dataset)
beta = [200, 400, 800, 1200, 1600, 2000, 2400, 3000]
rest_kv = [35.00, 35.19, 35.74, 35.65, 35.61, 35.89, 35.57, 35.39]

# Plotting
plt.figure(figsize=(8, 6))

colors = ['#2a9d8f']

# Plot each line with its specific style
plt.plot(beta, rest_kv, label="AWS", linestyle="-", marker="^", color=colors[0], linewidth=3, markersize=8)

# Add dashed horizontal line for "Full Cache"
plt.axhline(y=35.34, color='gray', linestyle='--', label='AvgPool')

# Add labels, title, and grid
plt.xlabel(r"Scale Factor $\beta$", fontsize=18)
plt.ylabel("Average Score", fontsize=18)
plt.title("Llama3.1-8B-Instruct", fontsize=24)
plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

plt.xticks(beta, fontsize=16)

y_ticks = [34, 34.5, 35, 35.5, 36]
plt.yticks(y_ticks, fontsize=16)

# Add legend
plt.legend(loc="lower right", fontsize=18, frameon=True, ncol=2, bbox_to_anchor=(1, 0.0))

# Show the plot
plt.tight_layout()
plt.savefig("figures/sensitivity_beta.pdf", dpi=300)
plt.show()

# %%
