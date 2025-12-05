import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from tqdm import tqdm
import os

from model import GraphTransformer, GraphTransformerSimple

def prepare_dataset(graph_path, temporal_cutoff=11.0, use_graph_masking=True):
    """
    Purpose: Load velocity-weighted graph and prepare for temporal holdout training
    Params: graph_path and temporal_cutoff
    Return: training data dictionary
    """
    # Load graph
    print(f"\nLoading graph from {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    
    if isinstance(graph, dict):
        # convert dict to Data object if needed
        graph = Data(
            x=graph['x'],
            edge_index=graph['edge_index'],
            edge_attr=graph.get('edge_attr', None),
            timepoint_norm=graph.get('timepoint_norm', None),
            celltype=graph['celltype'],
            celltype_names=graph.get('celltype_names', None),
        )
    
    # Extract data
    X = graph.x  # [N, 50] PCA features
    edge_index = graph.edge_index  # [2, E]
    edge_attr = graph.edge_attr  # [E, 1] velocity-weighted
    
    # Handle timepoint 
    if hasattr(graph, 'timepoint'):
        if isinstance(graph.timepoint, np.ndarray):
            timepoint = torch.from_numpy(graph.timepoint).float()
        else:
            timepoint = graph.timepoint.float()
    else:
        raise ValueError("Graph missing 'timepoint' attribute")
    
    # Handle celltype 
    if isinstance(graph.celltype, np.ndarray):
        labels = torch.from_numpy(graph.celltype).long()
    else:
        labels = graph.celltype.long()
    
    # Add names
    celltype_names = graph.celltype_names if hasattr(graph, 'celltype_names') else None
    
    print(f"\nDataset info:")
    print(f"Cells: {X.shape[0]:,}")
    print(f"Genes (PCA): {X.shape[1]}")
    print(f"Edges: {edge_index.shape[1]:,}")
    if celltype_names:
        print(f"Cell types: {celltype_names}")
    
    # Show timepoint distribution
    print(f"\nTimepoint distribution:")
    for t in sorted(timepoint.unique().tolist()):
        n_cells = (timepoint == t).sum().item()
        print(f"  E{t}: {n_cells:,} cells")
    
    # Create temporal split
    print(f"\nTemporal cutoff = E{temporal_cutoff}")
    
    train_mask = timepoint >= temporal_cutoff
    test_mask = timepoint < temporal_cutoff
    
    # val: use middle timepoint from training set
    train_timepoints = timepoint[train_mask].unique()
    if len(train_timepoints) > 2:
        val_timepoint = sorted(train_timepoints.tolist())[len(train_timepoints) // 2]
        val_mask = (timepoint == val_timepoint)
        train_mask = train_mask & ~val_mask
    else:
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    
    print(f"\nSplit summary:")
    print(f"TRAIN (E{temporal_cutoff}+): {train_mask.sum().item():,} cells (learn from differentiated)")
    if val_mask.sum() > 0:
        print(f"VAL: {val_mask.sum().item():,} cells")
    print(f"TEST (E<{temporal_cutoff}): {test_mask.sum().item():,} cells (predict progenitor fates)")
    
    # show label distribution in each split
    if celltype_names is not None:
        print(f"\nLabel distribution:")
        for split_name, mask in [('TRAIN', train_mask), ('VAL', val_mask), ('TEST', test_mask)]:
            if mask.sum() > 0:
                print(f"{split_name}:")
                for i, name in enumerate(celltype_names):
                    count = ((labels == i) & mask).sum().item()
                    pct = 100 * count / mask.sum().item() if mask.sum() > 0 else 0
                    print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Graph masking: filter edges to prevent information leakage
    train_edge_index = edge_index
    train_edge_attr = edge_attr
    val_edge_index = edge_index
    val_edge_attr = edge_attr
    test_edge_index = edge_index
    test_edge_attr = edge_attr
    
    if use_graph_masking:
        print(f"\n{'='*60}")
        print("GRAPH MASKING: Filtering edges to prevent information leakage")
        print(f"{'='*60}")
        total_edges = edge_index.shape[1]
        
        # Training: only train-train edges
        train_nodes = train_mask.nonzero(as_tuple=False).squeeze()
        train_edge_index, train_edge_attr = subgraph(
            subset=train_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_edge_mask=False
        )
        train_edges = train_edge_index.shape[1]
        print(f"Training graph: {total_edges:,} total edges -> {train_edges:,} train-train edges "
              f"({100*train_edges/total_edges:.1f}% retained)")
        
        # Validation: train+val edges only (exclude test nodes)
        if val_mask.sum() > 0:
            train_val_nodes = (train_mask | val_mask).nonzero(as_tuple=False).squeeze()
            val_edge_index, val_edge_attr = subgraph(
                subset=train_val_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_edge_mask=False
            )
            val_edges = val_edge_index.shape[1]
            print(f"Validation graph: {total_edges:,} total edges -> {val_edges:,} train+val edges "
                  f"({100*val_edges/total_edges:.1f}% retained)")
        
        # Test: train+test edges only (exclude val nodes)
        train_test_nodes = (train_mask | test_mask).nonzero(as_tuple=False).squeeze()
        test_edge_index, test_edge_attr = subgraph(
            subset=train_test_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_edge_mask=False
        )
        test_edges = test_edge_index.shape[1]
        print(f"Test graph: {total_edges:,} total edges -> {test_edges:,} train+test edges "
              f"({100*test_edges/total_edges:.1f}% retained)")
        print(f"{'='*60}\n")
    
    # Package data
    data = {
        'X': X,
        'train_edge_index': train_edge_index,  # Masked: train-train edges only
        'train_edge_attr': train_edge_attr,   # Masked: train-train edges only
        'val_edge_index': val_edge_index,      # Masked: train+val edges only
        'val_edge_attr': val_edge_attr,        # Masked: train+val edges only
        'test_edge_index': test_edge_index,    # Masked: train+test edges only
        'test_edge_attr': test_edge_attr,      # Masked: train+test edges only
        'timepoint': timepoint,
        'timepoint_norm': graph.timepoint_norm if hasattr(graph, 'timepoint_norm') else None,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_classes': len(celltype_names) if celltype_names else labels.max().item() + 1,
        'celltype_names': celltype_names,
        'temporal_cutoff': temporal_cutoff,
    }
    
    return data


def create_train_val_test_split(graph, temporal_cutoff=11.0):
    """
    Purpose: Create temporal holdout split for training
    Params: graph object and temporal_cutoff
    Return: graph with train_mask, val_mask, test_mask added
    """
    # Extract timepoint (handle both numpy and tensor)
    if hasattr(graph, 'timepoint'):
        if isinstance(graph.timepoint, np.ndarray):
            timepoint = torch.from_numpy(graph.timepoint).float()
        else:
            timepoint = graph.timepoint.float()
    else:
        raise ValueError("Graph missing 'timepoint' attribute")
    
    print(f"\nTemporal cutoff = E{temporal_cutoff}")
    
    # Create temporal split
    train_mask = timepoint >= temporal_cutoff
    test_mask = timepoint < temporal_cutoff
    
    # Validation: use middle timepoint from training set
    train_timepoints = timepoint[train_mask].unique()
    if len(train_timepoints) > 2:
        val_timepoint = sorted(train_timepoints.tolist())[len(train_timepoints) // 2]
        val_mask = (timepoint == val_timepoint)
        train_mask = train_mask & ~val_mask
    else:
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    
    print(f"\nSplit summary:")
    print(f"TRAIN (E{temporal_cutoff}+): {train_mask.sum().item():,} cells")
    if val_mask.sum() > 0:
        print(f"VAL: {val_mask.sum().item():,} cells")
    print(f"TEST (E<{temporal_cutoff}): {test_mask.sum().item():,} cells")
    
    # Show label distribution in each split
    if hasattr(graph, 'celltype_names') and graph.celltype_names is not None:
        print(f"\nLabel distribution:")
        for split_name, mask in [('TRAIN', train_mask), ('VAL', val_mask), ('TEST', test_mask)]:
            if mask.sum() > 0:
                print(f"{split_name}:")
                for i, name in enumerate(graph.celltype_names):
                    count = ((graph.celltype == i) & mask).sum().item()
                    pct = 100 * count / mask.sum().item() if mask.sum() > 0 else 0
                    print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Add masks to graph
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    
    return graph


def train_epoch(model, data, optimizer, criterion, device, scaler=None):
    model.train()
    optimizer.zero_grad()
    
    # Convert timepoint_norm to tensor if it's a numpy array
    timepoint_norm_tensor = None
    if data['timepoint_norm'] is not None:
        if isinstance(data['timepoint_norm'], np.ndarray):
            timepoint_norm_tensor = torch.from_numpy(data['timepoint_norm']).float().to(device)
        else:
            timepoint_norm_tensor = data['timepoint_norm'].to(device)
    
    # Move graph data to device - use masked training graph
    x = data['X'].to(device)
    edge_index = data['train_edge_index'].to(device)  # Use masked edges for training
    edge_attr = data['train_edge_attr'].to(device) if data['train_edge_attr'] is not None else None
    train_mask_device = data['train_mask'].to(device)
    
    # Use mixed precision for forward pass to reduce memory
    with autocast(device_type='cuda'):
        # forward pass
        if data['timepoint_norm'] is not None and data['train_edge_attr'] is not None:
            out = model(
                x,
                edge_index,
                edge_attr=edge_attr,
                timepoint_norm=timepoint_norm_tensor,
            )
        elif data['timepoint_norm'] is not None:
            out = model(
                x,
                edge_index,
                edge_attr=None,
                timepoint_norm=timepoint_norm_tensor,
            )
        else:
            out = model(x, edge_index)
        
        # compute loss only on training nodes
        celltype_device = data['labels'].to(device)
        loss = criterion(out[train_mask_device], celltype_device[train_mask_device])
    
    # backward pass with mixed precision
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    # compute accuracy (move to CPU to free GPU memory)
    pred = out[train_mask_device].argmax(dim=1).cpu()
    true = celltype_device[train_mask_device].cpu()
    acc = accuracy_score(true.numpy(), pred.numpy())
    
    # Clear cache
    del x, edge_index, out, train_mask_device, celltype_device
    if edge_attr is not None:
        del edge_attr
    if timepoint_norm_tensor is not None:
        del timepoint_norm_tensor
    torch.cuda.empty_cache()
    
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, mask, device, celltype_names=None, split='val'):
    """Evaluate model on a given mask.
    
    Args:
        split: 'val' or 'test' - determines which masked graph to use
    """
    model.eval()
    
    # Clear cache before evaluation
    torch.cuda.empty_cache()
    
    # Convert timepoint_norm to tensor if it's a numpy array
    timepoint_norm_tensor = None
    if data['timepoint_norm'] is not None:
        if isinstance(data['timepoint_norm'], np.ndarray):
            timepoint_norm_tensor = torch.from_numpy(data['timepoint_norm']).float().to(device)
        else:
            timepoint_norm_tensor = data['timepoint_norm'].to(device)
    
    # Move graph data to device - use masked graph for evaluation (prevents info leakage)
    x = data['X'].to(device)
    if split == 'val':
        edge_index = data['val_edge_index'].to(device)  # Masked: train+val edges only
        edge_attr = data['val_edge_attr'].to(device) if data['val_edge_attr'] is not None else None
    elif split == 'test':
        edge_index = data['test_edge_index'].to(device)  # Masked: train+test edges only
        edge_attr = data['test_edge_attr'].to(device) if data['test_edge_attr'] is not None else None
    else:
        raise ValueError(f"Unknown split: {split}")
    mask_device = mask.to(device)
    
    # Use mixed precision for evaluation to reduce memory
    with autocast(device_type='cuda'):
        if data['timepoint_norm'] is not None and edge_attr is not None:
            out = model(
                x,
                edge_index,
                edge_attr=edge_attr,
                timepoint_norm=timepoint_norm_tensor,
            )
        elif data['timepoint_norm'] is not None:
            out = model(
                x,
                edge_index,
                edge_attr=None,
                timepoint_norm=timepoint_norm_tensor,
            )
        else:
            out = model(x, edge_index)
    
    # Move predictions to CPU immediately to free GPU memory
    pred = out[mask_device].argmax(dim=1).cpu()
    true = data['labels'][mask].cpu()
    
    acc = accuracy_score(true.numpy(), pred.numpy())
    
    # Clear GPU memory
    del x, edge_index, out, mask_device
    if edge_attr is not None:
        del edge_attr
    if timepoint_norm_tensor is not None:
        del timepoint_norm_tensor
    torch.cuda.empty_cache()
    
    # classification metrics
    if celltype_names is not None:
        # Get unique labels present in the data
        unique_labels = np.unique(np.concatenate([true.numpy(), pred.numpy()]))
        # Filter target_names to only include labels that are present
        present_target_names = [celltype_names[i] for i in unique_labels if i < len(celltype_names)]
        report = classification_report(
            true.numpy(),
            pred.numpy(),
            labels=unique_labels,
            target_names=present_target_names,
            output_dict=True,
            zero_division=0
        )
    else:
        report = classification_report(
            true.numpy(),
            pred.numpy(),
            output_dict=True,
            zero_division=0
        )
    
    return acc, report, pred, true


def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Train Graph Transformer for cell fate classification')
    parser.add_argument('--graph_path', type=str, default='blood_graph_velocity.pt',
                        help='Path to preprocessed graph file')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GPSConv layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (usually 1 for single graph)')
    parser.add_argument('--use_timepoint', type=str_to_bool, required=True,
                        help='Whether to use timepoint as additional feature (true/false)')
    parser.add_argument('--use_edge_attr', type=str_to_bool, required=True,
                        help='Whether to use edge attributes (velocity-weighted) (true/false)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--temporal_cutoff', type=float, default=11.0,
                        help='Temporal cutoff for train/test split (E>=cutoff for train, E<cutoff for test)')
    parser.add_argument('--use_graph_masking', type=str_to_bool, default=True,
                        help='Whether to mask graph edges during training to prevent information leakage (true/false)')
    
    args = parser.parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear any leftover GPU memory from previous runs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # Print GPU memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            print(f"\nGPU Memory Status at Job Start:")
            print(f"  Total GPU Memory: {total:.2f} GB")
            print(f"  Already Allocated: {allocated:.2f} GB")
            print(f"  Already Reserved: {reserved:.2f} GB")
            print(f"  Free: {total - reserved:.2f} GB")
            if allocated > 0.1:
                print(f"  WARNING: GPU has {allocated:.2f} GB already allocated! This suggests leftover memory from a previous job.")
            torch.cuda.empty_cache()  # Try to clear again
            allocated_after = torch.cuda.memory_allocated(device) / 1024**3
            if allocated_after < allocated:
                print(f"  After clearing: {allocated_after:.2f} GB allocated")
    
    # Prepare dataset
    data = prepare_dataset(args.graph_path, temporal_cutoff=args.temporal_cutoff, 
                          use_graph_masking=args.use_graph_masking)
    
    # get num classes and input channels
    num_classes = data['num_classes']
    in_channels = data['X'].shape[1]
    
    print(f"\nModel configuration:")
    print(f"Input channels: {in_channels}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Output channels (classes): {num_classes}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Attention heads: {args.heads}")
    print(f"Dropout: {args.dropout}")
    
    # create model

    model = GraphTransformer(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=num_classes,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        use_timepoint=args.use_timepoint,
        use_edge_attr=args.use_edge_attr,
    ).to(device)

    print(f"\nModel settings:")
    print(f"  Edge attributes usage: {args.use_edge_attr}")
    print(f"  Timepoint features: {args.use_timepoint}")
    print(f"  Graph masking: {args.use_graph_masking}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize mixed precision scaler for memory efficiency
    # Use the new API to avoid deprecation warning
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training! 
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Using mixed precision training to reduce GPU memory usage")
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in tqdm(range(args.epochs), desc="Training"):
        # Clear cache at start of each epoch
        torch.cuda.empty_cache()
        
        #train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device, scaler)
        
        # val
        val_acc, val_report, _, _ = evaluate(model, data, data['val_mask'], device, 
                                             celltype_names=data['celltype_names'], split='val')
        
        # test
        test_acc, test_report, _, _ = evaluate(model, data, data['test_mask'], device,
                                                celltype_names=data['celltype_names'], split='test')
        
        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
        
        # print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Acc:   {val_acc:.4f}")
            print(f"  Test Acc:  {test_acc:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")
    
    # load best model and evaluate on test set
    print(f"\nLoading best model for final evaluation...")
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            # Try to load with strict=False first to see what matches
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys or unexpected_keys:
                print(f"Warning: Checkpoint has different architecture.")
                if missing_keys:
                    print(f"  Missing keys (not loaded): {len(missing_keys)} parameters")
                if unexpected_keys:
                    print(f"  Unexpected keys (ignored): {len(unexpected_keys)} parameters")
                print(f"  This usually happens when model architecture changed (e.g., hidden_channels or attention type).")
                print(f"  Using current model state (already trained) for final evaluation.")
            else:
                print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"Checkpoint validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {str(e)[:200]}")
            print(f"Using current model state for final evaluation.")
    else:
        print(f"Warning: Checkpoint file not found. Using current model state for final evaluation.")
    
    test_acc, test_report, test_pred, test_true = evaluate(
        model, data, data['test_mask'], device,
        celltype_names=data['celltype_names'], split='test'
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:")
    if data['celltype_names'] is not None:
        # Get unique labels present in the test data
        unique_labels = np.unique(np.concatenate([test_true.numpy(), test_pred.numpy()]))
        # Filter target_names to only include labels that are present
        present_target_names = [data['celltype_names'][i] for i in unique_labels if i < len(data['celltype_names'])]
        print(classification_report(
            test_true.numpy(),
            test_pred.numpy(),
            labels=unique_labels,
            target_names=present_target_names,
            zero_division=0
        ))
    else:
        print(classification_report(
            test_true.numpy(),
            test_pred.numpy(),
            zero_division=0
        ))


if __name__ == '__main__':
    main()

