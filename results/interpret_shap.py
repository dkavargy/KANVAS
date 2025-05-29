# === STEP 13: SHAP-style Class-Specific Feature Importance Plot (Stacked, Custom Colors) ===
class_0_mask = (y_sample == 0)  # Traditional
class_1_mask = (y_sample == 1)  # Modern

# Signed gradients
signed_grads = gradients * (2 * y_sample[:, np.newaxis] - 1)

# Average SHAP-style values
class_0_avg = np.abs(signed_grads[class_0_mask]).mean(axis=0)
class_1_avg = np.abs(signed_grads[class_1_mask]).mean(axis=0)

# Combined feature importance
combined_avg = class_0_avg + class_1_avg
top_feat_idx = np.argsort(combined_avg)[-20:][::-1]

# Custom colors
color_class_0 = '#008bf9'  # rgb(0,139,249)
color_class_1 = '#fe004f'  # Class 1

# Plot
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(top_feat_idx))

# Stacked bars
plt.barh(y_pos, class_0_avg[top_feat_idx], label="Modern (Class 0)", color=color_class_0)
plt.barh(y_pos, class_1_avg[top_feat_idx], left=class_0_avg[top_feat_idx], label="Traditional (Class 1)", color=color_class_1)

# Labels and styling
plt.yticks(y_pos, [skill_names[i] for i in top_feat_idx])
plt.xlabel("mean(|SHAP-style value|)")
plt.title("Class-Specific SHAP-style Feature Importance (Stacked)")
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True, axis='x', linestyle=':')
plt.show()
