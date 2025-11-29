import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/users/hanting/project/processed/ablation_results.csv")

# Convert config string to dict for easier access
import ast
df['config_dict'] = df['config'].apply(ast.literal_eval)
df['use_ode'] = df['config_dict'].apply(lambda x: x['use_ode'])
df['n_layers'] = df['config_dict'].apply(lambda x: x['n_layers'])
df['hid_dim'] = df['config_dict'].apply(lambda x: x['hid_dim'])

# Plot Top-1 accuracy vs hid_dim, grouped by n_layers and colored by use_ode
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='hid_dim', y='top1', hue='use_ode', style='n_layers', markers=True, dashes=False)
plt.title("Top-1 Accuracy vs Hidden Dimension")
plt.xlabel("Hidden Dimension")
plt.ylabel("Top-1 Accuracy")
plt.legend(title="Use ODE / n_layers")
plt.grid(True)
plt.savefig("/users/hanting/project/processed/plots/top1_vs_hiddim.png", dpi=150)
plt.show()

# Plot Macro F1 vs hid_dim, grouped similarly
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='hid_dim', y='f1', hue='use_ode', style='n_layers', markers=True, dashes=False)
plt.title("Macro F1 vs Hidden Dimension")
plt.xlabel("Hidden Dimension")
plt.ylabel("Macro F1 Score")
plt.legend(title="Use ODE / n_layers")
plt.grid(True)
plt.savefig("/users/hanting/project/processed/plots/f1_vs_hiddim.png", dpi=150)
plt.show()

# Heatmap of top1 for use_ode=True only, by n_layers and hid_dim
pivot_table = df[df['use_ode']].pivot(index='n_layers', columns='hid_dim', values='top1')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='YlGnBu')
plt.title("Top-1 Accuracy Heatmap (use_ode=True)")
plt.savefig("/users/hanting/project/processed/plots/top1_heatmap_ode_true.png", dpi=150)
plt.show()
