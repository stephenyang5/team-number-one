import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from tqdm import tqdm
import os

from model import GraphTransformer, GraphTransformerSimple

def load_graph(graph_path):
    # load a graph
    print(f"Loading graph from {graph_path}...")
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
    
    print(f"Graph loaded:")
    print(f"Nodes: {graph.x.shape[0]:,}")
    print(f"Edges: {graph.edge_index.shape[1]:,}")
    print(f"Node features: {graph.x.shape[1]}")
    print(f"Cell types: {len(graph.celltype_names) if hasattr(graph, 'celltype_names') else 'unknown'}")
    
    return graph


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


def train_epoch(model, graph, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    # Convert timepoint_norm to tensor if it's a numpy array
    timepoint_norm_tensor = None
    if hasattr(graph, 'timepoint_norm') and graph.timepoint_norm is not None:
        if isinstance(graph.timepoint_norm, np.ndarray):
            timepoint_norm_tensor = torch.from_numpy(graph.timepoint_norm).float().to(device)
        else:
            timepoint_norm_tensor = graph.timepoint_norm.to(device)
    
    # Move graph data to device
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
    
    # forward pass
    if hasattr(graph, 'timepoint_norm') and hasattr(graph, 'edge_attr'):
        out = model(
            x,
            edge_index,
            edge_attr=edge_attr,
            timepoint_norm=timepoint_norm_tensor,
        )
    elif hasattr(graph, 'timepoint_norm'):
        out = model(
            x,
            edge_index,
            edge_attr=None,
            timepoint_norm=timepoint_norm_tensor,
        )
    else:
        out = model(x, edge_index)
    
    # compute loss only on training nodes
    train_mask_device = graph.train_mask.to(device)
    celltype_device = graph.celltype.to(device)
    loss = criterion(out[train_mask_device], celltype_device[train_mask_device])
    
    # backward pass
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
def evaluate(model, graph, mask, device, celltype_names=None):
    """Evaluate model on a given mask."""
    model.eval()
    
    # Clear cache before evaluation
    torch.cuda.empty_cache()
    
    # Convert timepoint_norm to tensor if it's a numpy array
    timepoint_norm_tensor = None
    if hasattr(graph, 'timepoint_norm') and graph.timepoint_norm is not None:
        if isinstance(graph.timepoint_norm, np.ndarray):
            timepoint_norm_tensor = torch.from_numpy(graph.timepoint_norm).float().to(device)
        else:
            timepoint_norm_tensor = graph.timepoint_norm.to(device)
    
    # Move graph data to device
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
    mask_device = mask.to(device)
    
    if hasattr(graph, 'timepoint_norm') and hasattr(graph, 'edge_attr'):
        out = model(
            x,
            edge_index,
            edge_attr=edge_attr,
            timepoint_norm=timepoint_norm_tensor,
        )
    elif hasattr(graph, 'timepoint_norm'):
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
    true = graph.celltype[mask].cpu()
    
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
    parser.add_argument('--use_timepoint', action='store_false', default=False,
                        help='Do not use timepoint as additional feature')
    parser.add_argument('--use_edge_attr', action='store_true', default=True,
                        help='Use edge attributes (velocity-weighted)')
    parser.add_argument('--simple', action='store_true',
                        help='Use simplified model (no timepoint/edge_attr)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--temporal_cutoff', type=float, default=11.0,
                        help='Temporal cutoff for train/test split (E>=cutoff for train, E<cutoff for test)')
    
    args = parser.parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load graph
    graph = load_graph(args.graph_path)
    
    # get num classes
    num_classes = len(graph.celltype_names) if hasattr(graph, 'celltype_names') else graph.celltype.max().item() + 1
    in_channels = graph.x.shape[1]
    
    print(f"\nModel configuration:")
    print(f"Input channels: {in_channels}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Output channels (classes): {num_classes}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Attention heads: {args.heads}")
    print(f"Dropout: {args.dropout}")
    
    # create train/val/test split
    graph = create_train_val_test_split(graph, temporal_cutoff=args.temporal_cutoff)
    
    # create model
    if args.simple:
        model = GraphTransformerSimple(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=num_classes,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
        ).to(device)
    else:
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
    
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training! 
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in tqdm(range(args.epochs), desc="Training"):
        # Clear cache at start of each epoch
        torch.cuda.empty_cache()
        
        #train
        train_loss, train_acc = train_epoch(model, graph, optimizer, criterion, device)
        
        # val
        val_acc, val_report, _, _ = evaluate(model, graph, graph.val_mask, device, 
                                             celltype_names=graph.celltype_names if hasattr(graph, 'celltype_names') else None)
        
        # test
        test_acc, test_report, _, _ = evaluate(model, graph, graph.test_mask, device,
                                                celltype_names=graph.celltype_names if hasattr(graph, 'celltype_names') else None)
        
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
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_report, test_pred, test_true = evaluate(
        model, graph, graph.test_mask, device,
        celltype_names=graph.celltype_names if hasattr(graph, 'celltype_names') else None
    )
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:")
    if hasattr(graph, 'celltype_names'):
        # Get unique labels present in the test data
        unique_labels = np.unique(np.concatenate([test_true.numpy(), test_pred.numpy()]))
        # Filter target_names to only include labels that are present
        present_target_names = [graph.celltype_names[i] for i in unique_labels if i < len(graph.celltype_names)]
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

