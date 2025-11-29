import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, top_k_accuracy_score
from umap import UMAP
from gtnode_model import GTNode

def evaluate_model(
    processed_npz_path="/users/hanting/project/processed/processed_data.npz",
    model_path="/users/hanting/project/processed/gtnode_best.pth",
    save_umap_path="/users/hanting/project/processed/latent_umap.png",
):
    data = np.load(processed_npz_path, allow_pickle=True)
    X = torch.from_numpy(data["features"]).float()
    labels = data["labels"]
    num_classes = int(data["num_classes"][0])

    rows = data["adj_row"].astype(np.int64)
    cols = data["adj_col"].astype(np.int64)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_weight = torch.from_numpy(data["adj_data"].astype(np.float32)).unsqueeze(1)  # [num_edges, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GTNode(in_dim=X.shape[1], hid_dim=128, n_layers=3, heads=4, use_ode=True, ode_steps=5, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    X = X.to(device)

    with torch.no_grad():
        logits, hT = model(X, edge_index, edge_weight)
        probs = torch.softmax(logits.cpu(), dim=1).numpy()
        preds = probs.argmax(axis=1)

    print("Top-1 acc", (preds == labels).mean())
    print("Top-3 acc", top_k_accuracy_score(labels, probs, k=3))
    print("Macro F1", f1_score(labels, preds, average="macro"))

    # UMAP of latent embeddings
    h = hT.cpu().numpy()
    um = UMAP(n_components=2, random_state=0).fit_transform(h)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=um[:, 0], y=um[:, 1], hue=labels, s=5, palette="tab20")
    plt.title("GTNode latent UMAP")
    plt.savefig(save_umap_path, dpi=150)
    print(f"Saved {save_umap_path}")

if __name__ == "__main__":
    evaluate_model()
