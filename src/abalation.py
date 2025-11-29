import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score, top_k_accuracy_score
from gtnode_model import GTNode

# Load processed data (base)
def load_data(npz_path, n_neighbors=None):
    data = np.load(npz_path, allow_pickle=True)
    # Load saved arrays
    features_np = data["features"].astype(np.float32)
    rows = data["adj_row"].astype(np.int64)
    cols = data["adj_col"].astype(np.int64)
    vals_np = data["adj_data"].astype(np.float32)
    labels_np = data["labels"].astype(np.int64)
    num_classes = int(data["num_classes"][0])
    
    
    # Construct PyG-style tensors
    edge_index_np = np.vstack([rows, cols])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(vals_np, dtype=torch.float32).view(-1, 1)
    X = torch.tensor(features_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    
    return X, edge_index, edge_attr, labels, num_classes

def train_and_evaluate(config, X, edge_index, edge_attr, labels, num_classes, device):
    model = GTNode(in_dim=X.shape[1],
                   hid_dim=config['hid_dim'],
                   n_layers=config['n_layers'],
                   heads=4,
                   use_ode=config['use_ode'],
                   ode_steps=5,
                   num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    N = X.shape[0]
    indices = torch.randperm(N)
    train_idx = indices[:int(0.7 * N)]
    val_idx = indices[int(0.7 * N): int(0.85 * N)]
    test_idx = indices[int(0.85 * N):]

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, 51): 
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X, edge_index, edge_attr)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(X, edge_index, edge_attr)
            preds = logits.argmax(dim=1)
            acc_train = (preds[train_idx] == labels[train_idx]).float().mean().item()
            acc_val = (preds[val_idx] == labels[val_idx]).float().mean().item()
            acc_test = (preds[test_idx] == labels[test_idx]).float().mean().item()

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_state = model.state_dict()

    # Load best model and evaluate final metrics
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(X, edge_index, edge_attr)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        labels_np = labels.cpu().numpy()

        top1 = (preds == labels_np).mean()
        top3 = top_k_accuracy_score(labels_np, probs, k=3)
        f1 = f1_score(labels_np, preds, average='macro')

    return top1, top3, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_path = "/users/hanting/project/processed/processed_data.npz"

    # Ablation configs
    use_ode_options = [False, True]
    n_layers_options = [2, 3, 5]
    hid_dim_options = [64, 128, 256]

    results = []
    for use_ode in use_ode_options:
        for n_layers in n_layers_options:
            for hid_dim in hid_dim_options:
                config = {
                    'use_ode': use_ode,
                    'n_layers': n_layers,
                    'hid_dim': hid_dim
                }

                # Load data 
                X, edge_index, edge_attr, labels, num_classes = load_data(npz_path)

                # Move tensors to device
                X = X.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                labels = labels.to(device)

                print(f"Training with config: {config}")
                top1, top3, f1 = train_and_evaluate(config, X, edge_index, edge_attr, labels, num_classes, device)
                print(f"Results: Top1 {top1:.4f}, Top3 {top3:.4f}, F1 {f1:.4f}")

                results.append({
                    'config': str(config),
                    'top1': top1,
                    'top3': top3,
                    'f1': f1,
                    'notes': ''
                })

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("/users/hanting/project/processed/ablation_results.csv", index=False)
    print("Saved ablation_results.csv")

if __name__ == "__main__":
    main()
