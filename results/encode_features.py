import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Enhanced SHAP-like Beeswarm Plot ===
plt.figure(figsize=(12, 10))

for idx, feature_idx in enumerate(top_indices):
    values = X_sample[:, feature_idx]
    shap_values = gradients[:, feature_idx] * (2 * y_sample - 1)  # signed gradient
    normed_vals = (values - values.min()) / (values.max() - values.min() + 1e-8)
    colors = cm.coolwarm(normed_vals)  # better perceptual balance than bwr

    # Spread vertically for readability
    jitter = np.random.normal(loc=0, scale=0.12, size=len(shap_values))
    y_positions = np.full_like(shap_values, idx, dtype=float) + jitter

    plt.scatter(shap_values, y_positions,
                alpha=0.7, c=colors, edgecolor='k', linewidth=0.25, s=40)

# Y-axis: skill names
plt.yticks(np.arange(len(top_indices)), [skill_names[i] for i in top_indices])

# Styling
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("SHAP-style Impact on Model Output")
plt.title("🎯 Enhanced SHAP-like Beeswarm Plot for Top 35 Skills", fontsize=14)
plt.grid(True, axis='x', linestyle=':', alpha=0.6)
plt.xlim(-3, 3)  # optional: tighter x-axis
plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap='coolwarm'), label='Feature Value')
plt.tight_layout()
plt.show()
