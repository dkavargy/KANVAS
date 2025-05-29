import shap
import matplotlib.pyplot as plt

# Select a few samples
num_samples = 200
X_decision = X_sample[:num_samples]
grads_decision = gradients[:num_samples]

# Compute SHAP-style values: gradient * input
shap_values_decision = grads_decision * X_decision
expected_value = model(X_train_tensor.to(device)).mean().item()

# Plot decision plot (manual input)
shap.plots.decision(
    base_value=expected_value,
    shap_values=shap_values_decision,
    feature_names=skill_names,
    feature_order="importance"
)

##################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Filter modern jobs
modern_mask = (y_sample == 1)
X_modern = X_sample[modern_mask]
gradients_modern = gradients[modern_mask]

# Step 2: Compute SHAP-style values (gradient × input)
shap_values_modern = gradients_modern * X_modern

# Step 3: Create DataFrame for seaborn boxplot
df_shap = pd.DataFrame(shap_values_modern, columns=skill_names)

# Optional: Show only top N features by mean absolute SHAP
top_n = 25
top_features = df_shap.abs().mean().sort_values(ascending=False).head(top_n).index
df_shap_top = df_shap[top_features]

# Step 4: Melt DataFrame into long format for seaborn
df_melt = df_shap_top.melt(var_name='Skill', value_name='SHAP Value')

# Step 5: Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melt, x='Skill', y='SHAP Value')
plt.xticks(rotation=90)
plt.title('SHAP Value Distribution for Modern Jobs (Top 25 Skills)')
plt.tight_layout()
plt.show()

##################

import shap
import matplotlib.pyplot as plt

# Optional: use a subset for clarity
X_heatmap = X_sample[:800]
shap_heat_values = gradients[:500] * (2 * y_sample[:500].reshape(-1, 1) - 1)  # simulate SHAP influence

# Create SHAP Explanation object
explainer = shap.Explanation(
    values=shap_heat_values,
    data=X_heatmap,
    feature_names=skill_names
)

# Plot heatmap
shap.plots.heatmap(explainer, max_display=20, show=True)

