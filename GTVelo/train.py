import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path to allow importing GTVelo
sys.path.insert(0, str(Path(__file__).parent.parent))
from GTVelo.model_transformer import create_model

"""Training setup for one epoch"""
def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    
    # Move data to device
    X = data['X'].to(device)
    edge_index = data['edge_index'].to(device)
    edge_attr = data['edge_attr'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    
    # Forward pass on ALL cells 
    optimizer.zero_grad()
    logits, _ = model(X, edge_index, edge_attr)
    
    # Loss computed ONLY on training cells (late timepoints)
    loss = criterion(logits[train_mask], labels[train_mask])
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Training accuracy
    with torch.no_grad():
        pred = logits[train_mask].argmax(dim=1)
        train_acc = (pred == labels[train_mask]).float().mean().item()
    
    return loss.item(), train_acc

"""Evaluate on validation or test set"""
@torch.no_grad()
def evaluate(model, data, criterion, device, split='val'):
    model.eval()
    
    # move data to device
    X = data['X'].to(device)
    edge_index = data['edge_index'].to(device)
    edge_attr = data['edge_attr'].to(device)
    labels = data['labels'].to(device)
    
    # Get appropriate mask
    if split == 'val':
        mask = data['val_mask'].to(device)
    else:  # test
        mask = data['test_mask'].to(device)
    
    if mask.sum() == 0:
        return 0.0, 0.0
    
    # Forward pass on ALL cells
    logits, _ = model(X, edge_index, edge_attr)
    
    # Evaluate only on specified split
    loss = criterion(logits[mask], labels[mask])
    pred = logits[mask].argmax(dim=1)
    acc = (pred == labels[mask]).float().mean().item()
    
    return loss.item(), acc

"""
Purpose: Main training loop
Params: data, model configuration, and training configuration
Return: trained model, testing accuracy
"""
def train(data, model_config, train_config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Begin training setup")
    print(f"Device: {device}")
    print(f"Temporal cutoff: E{data['temporal_cutoff']}")
    print(f"Training cells: {data['train_mask'].sum():,} (late timepoints)")
    print(f"Test cells: {data['test_mask'].sum():,} (progenitors)")
    
    # Create model
    model = create_model(
        num_classes=data['num_classes'],
        **model_config
    ).to(device)
    
    # Optimizer and loss...
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay']
    )
    
    # Class-balanced loss to handle imbalance
    # Calculate class weights (inverse frequency)
    train_labels = data['labels'][data['train_mask']]
    class_counts = torch.bincount(train_labels, minlength=data['num_classes'])
    
    # Avoid division by zero and compute weights
    class_weights = torch.zeros(data['num_classes'])
    for i in range(data['num_classes']):
        if class_counts[i] > 0:
            class_weights[i] = len(train_labels) / (data['num_classes'] * class_counts[i])
        else:
            class_weights[i] = 0.0  # No samples, so weight is 0
    
    print(f"\nClass distribution and weights:")
    for i, name in enumerate(data['celltype_names']):
        if class_counts[i] > 0:
            print(f"  {name:30} weight: {class_weights[i]:.4f} ({class_counts[i]:,} train samples)")
        else:
            print(f"  {name:30} weight: 0.0000 (NOT IN TRAINING SET)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Learning rate scheduler 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print("\n" + "-" * 50)
    print("TRAINING")
    
    # Progress bar for epochs
    pbar = tqdm(range(1, train_config['epochs'] + 1), desc="Training", ncols=100)
    
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, data, criterion, device, split='val')
        
        # Update learning rate
        if val_acc > 0:  # Only if we have validation set
            scheduler.step(val_acc)
        else:
            scheduler.step(train_acc)
        
        # Early stopping
        current_metric = val_acc if val_acc > 0 else train_acc
        if current_metric > best_val_acc:
            best_val_acc = current_metric
            patience_counter = 0
            
            # Save best model
            save_dir = Path(train_config['save_dir'])
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / 'gtvelo_best.pt'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': model_config,
            }, save_path)
        else:
            patience_counter += 1
        
        # Update progress bar
        val_str = f"Val: {val_acc:.3f}" if val_acc > 0 else "Val: N/A"
        pbar.set_postfix({
            'Loss': f'{train_loss:.4f}',
            'Train': f'{train_acc:.3f}',
            'Val': val_str,
            'Best': f'{best_val_acc:.3f}',
            'Patience': f'{patience_counter}/{train_config["patience"]}'
        })
        
        # Detailed logging at intervals
        if epoch % train_config['log_interval'] == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"Epoch {epoch:03d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"{val_str} | "
                      f"Best: {best_val_acc:.4f} | "
                      f"LR: {lr:.2e}")
        
        # Early stopping
        if patience_counter >= train_config['patience']:
            tqdm.write(f"\nEarly stopping at epoch {epoch}")
            break
    
    pbar.close()
    
    # Load best model and evaluate on test set
    print("\n" + "-" * 50)
    print("FINAL EVALUATION ON PROGENITORS (TEST SET)")
    
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, data, criterion, device, split='test')
    
    print(f"\n✓ Test Accuracy (Progenitor Fate Prediction): {test_acc:.4f}")
    print(f"  ({data['test_mask'].sum():,} progenitor cells)")
    
    # Per-class accuracy
    print(f"\nPer-class performance on test set:")
    model.eval()
    with torch.no_grad():
        X = data['X'].to(device)
        edge_index = data['edge_index'].to(device)
        edge_attr = data['edge_attr'].to(device)
        labels = data['labels'].to(device)
        test_mask = data['test_mask'].to(device)
        
        logits, _ = model(X, edge_index, edge_attr)
        pred = logits[test_mask].argmax(dim=1)
        
        for i, name in enumerate(data['celltype_names']):
            class_mask = labels[test_mask] == i
            if class_mask.sum() > 0:
                class_acc = (pred[class_mask] == labels[test_mask][class_mask]).float().mean().item()
                n_samples = class_mask.sum().item()
                print(f"  {name:30} {class_acc:.4f} ({n_samples} cells)")
    
    return model, test_acc


if __name__ == "__main__":
    # Load prepared data
    print("Loading prepared data...")
    data = torch.load('neuron/neuron_prepared_velocity.pt') #change depencing on dataset
    
    # Model configuration 
    model_config = {
        'in_dim': 50,
        'hid_dim': 128,  
        'n_layers': 3,  
        'heads': 4,    
        'dropout': 0.2,    #prev 0.3
    }
    
    # Training configuration
    train_config = {
        'lr': 1e-3,  #prev 1e^-4
        'weight_decay': 1e-4,
        'epochs': 200,
        'patience': 50, 
        'log_interval': 5,
        'save_dir': 'checkpoints',
    }
    
    # Train
    model, test_acc = train(data, model_config, train_config)
    
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    