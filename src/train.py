import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gtnode_model import GTNode

def train_model(
    processed_npz_path="/users/hanting/project/processed/processed_data.npz",
    use_ode=True,
    n_layers=3,
    hid_dim=128,
    heads=4,
    ode_steps=5,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=100,
    save_path="/users/hanting/project/processed/gtnode_best.pth",
):
    data = np.load(processed_npz_path, allow_pickle=True)
    features_np = data["features"].astype(np.float32)           # [N, d]
    rows = data["adj_row"].astype(np.int64)
    cols = data["adj_col"].astype(np.int64)
    vals_np = data["adj_data"].astype(np.float32)
    labels_np = data["labels"].astype(np.int64)
    num_classes = int(data["num_classes"][0])

    edge_index_np = np.vstack([rows, cols])  # shape [2, E]
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    edge_attr = torch.tensor(vals_np, dtype=torch.float32).view(-1, 1)

    X = torch.tensor(features_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    labels = labels.to(device)

    model = GTNode(
        in_dim=X.shape[1],
        hid_dim=hid_dim,
        n_layers=n_layers,
        heads=heads,
        use_ode=use_ode,
        ode_steps=ode_steps,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    N = X.shape[0]
    indices = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx = indices[: int(0.7 * N)]
    val_idx = indices[int(0.7 * N): int(0.85 * N)]
    test_idx = indices[int(0.85 * N):]

    train_idx = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_idx = torch.tensor(val_idx, dtype=torch.long).to(device)
    test_idx = torch.tensor(test_idx, dtype=torch.long).to(device)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X, edge_index, edge_attr)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(X, edge_index, edge_attr)
            pred = logits.argmax(dim=1)
            acc_train = (pred[train_idx] == labels[train_idx]).float().mean().item()
            acc_val = (pred[val_idx] == labels[val_idx]).float().mean().item()
            acc_test = (pred[test_idx] == labels[test_idx]).float().mean().item()

        if acc_val > best_val:
            best_val = acc_val
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} Loss {loss.item():.4f} Train {acc_train:.4f} Val {acc_val:.4f} Test {acc_test:.4f}")

    return save_path
