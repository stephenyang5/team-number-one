import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from torch_geometric.explain import Explainer
    from torch_geometric.explain.algorithm import AttentionExplainer
    PYG_EXPLAIN_AVAILABLE = True
except ImportError:
    PYG_EXPLAIN_AVAILABLE = False
    print("Warning: torch_geometric.explain not available. Using manual attention extraction.")


def extract_attention_weights(model, x, edge_index, edge_attr=None, device='cpu'):
    """
    Extract attention weights from all TransformerConv layers in the model.
    
    Args:
        x: node features [N, in_dim]
        edge_index: edge indices [2, E]
        edge_attr: edge attributes [E, 1]
        device: Device to run on
    
    Returns:
        List of attention weights for each layer, each as (edge_index, attention_weights)
        where attention_weights shape is [E, heads] for multi-head attention
    """
    model.eval()
    x = x.to(device)
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    
    all_attention_weights = []
    
    # forward through input layer
    x = model.encoder.input_lin(x)
    x = torch.relu(x)
    
    # extract attention from each transformer layer
    for i, (conv, norm) in enumerate(zip(model.encoder.layers, model.encoder.norms)):
        x_res = x
        
        # forward with attention weights
        x_out, (edge_idx, attn_weights) = conv(
            x, edge_index, edge_attr=edge_attr, 
            return_attention_weights=True
        )
        
        # attn_weights shape: [E, heads] for multi-head attention
        all_attention_weights.append({
            'layer': i,
            'edge_index': edge_idx.cpu(),
            'attention': attn_weights.cpu(),  # [E, heads]
            'edge_attr': edge_attr.cpu() if edge_attr is not None else None
        })
        
        x = norm(x_out)
        x = torch.relu(x)
        x = x + x_res
        x = model.encoder.dropout(x)
    
    return all_attention_weights


def aggregate_attention_weights(attention_weights_list, method='mean'):
    """
    Aggregate attention weights across layers and heads.
    
    Args:
        attention_weights_list: List of attention weight dicts from extract_attention_weights

    Returns:
        Dictionary with aggregated attention per edge
    """
    # Build edge -> attention mapping
    edge_attention_map = defaultdict(list)
    
    for layer_data in attention_weights_list:
        edge_idx = layer_data['edge_index']  # [2, E]
        attn = layer_data['attention']  # [E, heads]
        
        # Average across heads for this layer
        if attn.dim() == 2:
            attn_per_edge = attn.mean(dim=1)  # [E]
        else:
            attn_per_edge = attn
        
        # Store attention for each edge
        for e_idx in range(edge_idx.shape[1]):
            src, dst = edge_idx[0, e_idx].item(), edge_idx[1, e_idx].item()
            edge_key = (src, dst)
            edge_attention_map[edge_key].append(attn_per_edge[e_idx].item())
    
    # Aggregate across layers
    aggregated = {}
    for edge_key, attn_values in edge_attention_map.items():
        if method == 'mean':
            aggregated[edge_key] = np.mean(attn_values)
        elif method == 'max':
            aggregated[edge_key] = np.max(attn_values)
        elif method == 'sum':
            aggregated[edge_key] = np.sum(attn_values)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregated


def visualize_attention_subgraph(
    data,
    attention_weights,
    node_idx: int,
    top_k: int = 20,
    celltype_names: Optional[List[str]] = None,
    timepoint: Optional[torch.Tensor] = None,
    exclude_celltypes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize attention for a specific cell (node) and its top-k most attended neighbors.
    
    Args:
        data: data dict with X, edge_index, labels, etc.
        attention_weights: aggregated attention weights dict
        node_idx: index of the cell to visualize
        top_k: number of top neighbors to show
        celltype_names: list of cell type names
        timepoint: timepoint tensor for coloring
        exclude_celltypes: list of cell type names to exclude from neighbors (e.g., ['Primitive erythroid cells'])
        save_path: path to save figure
        figsize: figure size
    """
    # Get labels and celltype names
    labels = data['labels']
    if celltype_names is None:
        celltype_names = data.get('celltype_names', [f'Type{i}' for i in range(data['num_classes'])])
    
    # Build set of excluded cell type indices
    excluded_type_indices = set()
    if exclude_celltypes:
        for exclude_name in exclude_celltypes:
            try:
                exclude_idx = celltype_names.index(exclude_name)
                excluded_type_indices.add(exclude_idx)
            except ValueError:
                print(f"Warning: Cell type '{exclude_name}' not found in celltype_names. Ignoring.")
    
    # Find edges involving this node
    node_edges = [(src, dst, attn) for (src, dst), attn in attention_weights.items() 
                  if src == node_idx or dst == node_idx]
    
    if not node_edges:
        print(f"No edges found for node {node_idx}")
        return
    
    # Filter out edges where the neighbor (not the query node) is an excluded cell type
    if excluded_type_indices:
        filtered_edges = []
        for src, dst, attn in node_edges:
            # Determine which node is the neighbor (not the query node)
            neighbor = dst if src == node_idx else src
            neighbor_type = labels[neighbor].item()
            
            # Only include if neighbor is not in excluded types
            if neighbor_type not in excluded_type_indices:
                filtered_edges.append((src, dst, attn))
        
        node_edges = filtered_edges
        
        if not node_edges:
            print(f"No edges found for node {node_idx} after excluding cell types: {exclude_celltypes}")
            return
    
    # Sort by attention and get top-k
    node_edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = node_edges[:top_k]
    
    # Build subgraph
    nodes_in_subgraph = {node_idx}
    for src, dst, _ in top_edges:
        nodes_in_subgraph.add(src)
        nodes_in_subgraph.add(dst)
    
    nodes_in_subgraph = sorted(list(nodes_in_subgraph))
    node_to_idx = {n: i for i, n in enumerate(nodes_in_subgraph)}
    
    # Create NetworkX graph
    G = nx.DiGraph()
    edge_weights = {}
    
    for src, dst, attn in top_edges:
        G.add_edge(node_to_idx[src], node_to_idx[dst])
        edge_weights[(node_to_idx[src], node_to_idx[dst])] = attn
    
    # Get node attributes (already have labels and celltype_names from above)
    node_colors = []
    node_labels = {}
    for n in nodes_in_subgraph:
        celltype = labels[n].item()
        node_colors.append(celltype)
        node_labels[node_to_idx[n]] = celltype_names[celltype] if celltype < len(celltype_names) else f'Type{celltype}'
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges with attention weights as width/color
    max_attn = max(edge_weights.values()) if edge_weights else 1.0
    edges = G.edges()
    edge_colors = [edge_weights.get((u, v), 0) / max_attn for u, v in edges]
    edge_widths = [3 * edge_weights.get((u, v), 0) / max_attn for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos, ax=ax, 
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # Draw nodes colored by cell type
    unique_types = sorted(set(node_colors))
    cmap = plt.cm.get_cmap('tab10', len(unique_types))
    node_color_map = [cmap(unique_types.index(c)) for c in node_colors]
    
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_color_map,
        node_size=500,
        alpha=0.8
    )
    
    # Highlight the query node
    query_node_idx = node_to_idx[node_idx]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=[query_node_idx],
        node_color='yellow',
        node_size=800,
        alpha=1.0,
        edgecolors='black',
        linewidths=3
    )
    
    # Add labels
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=8)
    
    # Add legend
    patches = [mpatches.Patch(color=cmap(i), label=celltype_names[t]) 
               for i, t in enumerate(unique_types)]
    ax.legend(handles=patches, loc='upper right')
    
    title = f'Attention Visualization for Cell {node_idx}\nTop-{top_k} Most Attended Neighbors'
    if exclude_celltypes:
        title += f'\n(Excluding: {", ".join(exclude_celltypes)})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()


def visualize_attention_heatmap(
    attention_weights,
    edge_index,
    celltype_names: Optional[List[str]] = None,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Create a heatmap showing attention weights between cell types.
    
    Args:
        attention_weights: aggregated attention weights dict
        edge_index: edge indices [2, E]
        celltype_names: list of cell type names
        labels: cell type labels [N]
        save_path: path to save figure
        figsize: figure size
    """
    if labels is None:
        raise ValueError("labels required for attention heatmap")
    
    num_types = len(celltype_names) if celltype_names else labels.max().item() + 1
    
    # Build type-to-type attention matrix
    type_attention = np.zeros((num_types, num_types))
    type_counts = np.zeros((num_types, num_types))
    
    edge_idx_np = edge_index.cpu().numpy()
    for e_idx in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, e_idx], edge_idx_np[1, e_idx]
        src_type = labels[src].item()
        dst_type = labels[dst].item()
        
        edge_key = (int(src), int(dst))
        if edge_key in attention_weights:
            attn = attention_weights[edge_key]
            type_attention[src_type, dst_type] += attn
            type_counts[src_type, dst_type] += 1
    
    # Average attention
    type_attention = np.divide(
        type_attention, 
        type_counts, 
        out=np.zeros_like(type_attention), 
        where=type_counts != 0
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(type_attention, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    if celltype_names:
        ax.set_xticks(range(num_types))
        ax.set_yticks(range(num_types))
        ax.set_xticklabels(celltype_names, rotation=45, ha='right')
        ax.set_yticklabels(celltype_names)
    else:
        ax.set_xticks(range(num_types))
        ax.set_yticks(range(num_types))
        ax.set_xticklabels([f'Type{i}' for i in range(num_types)], rotation=45, ha='right')
        ax.set_yticklabels([f'Type{i}' for i in range(num_types)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Attention Weight', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(num_types):
        for j in range(num_types):
            if type_counts[i, j] > 0:
                text = ax.text(j, i, f'{type_attention[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel('Target Cell Type', fontsize=12)
    ax.set_ylabel('Source Cell Type', fontsize=12)
    ax.set_title('Attention Weights Between Cell Types', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    else:
        plt.show()


def visualize_node_attention_scores(
    attention_weights,
    edge_index,
    num_nodes: int,
    celltype_names: Optional[List[str]] = None,
    labels: Optional[torch.Tensor] = None,
    top_k: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    visualize which nodes receive the most attention (incoming + outgoing).
    
    Args:
        attention_weights: Aggregated attention weights dict
        edge_index: Edge indices [2, E]
        num_nodes: Number of nodes
        celltype_names: List of cell type names
        labels: Cell type labels
        top_k: Number of top nodes to show
        save_path: Path to save figure
        figsize: Figure size
    """
    # Compute total attention per node (incoming + outgoing)
    node_attention = np.zeros(num_nodes)
    
    edge_idx_np = edge_index.cpu().numpy()
    for e_idx in range(edge_idx_np.shape[1]):
        src, dst = edge_idx_np[0, e_idx], edge_idx_np[1, e_idx]
        
        edge_key = (int(src), int(dst))
        if edge_key in attention_weights:
            attn = attention_weights[edge_key]
            node_attention[src] += attn  # Outgoing attention
            node_attention[dst] += attn  # Incoming attention
    
    # Get top-k nodes
    top_indices = np.argsort(node_attention)[-top_k:][::-1]
    top_scores = node_attention[top_indices]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by cell type if available
    if labels is not None and celltype_names is not None:
        top_labels = [labels[i].item() for i in top_indices]
        unique_types = sorted(set(top_labels))
        cmap = plt.cm.get_cmap('tab10', len(unique_types))
        colors = [cmap(unique_types.index(l)) for l in top_labels]
    else:
        colors = 'steelblue'
    
    bars = ax.barh(range(len(top_indices)), top_scores, color=colors)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f'Cell {i}' for i in top_indices])
    ax.set_xlabel('Total Attention Score', fontsize=12)
    ax.set_ylabel('Cell Index', fontsize=12)
    ax.set_title(f'Top-{top_k} Cells by Attention Score', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add legend if using cell type colors
    if labels is not None and celltype_names is not None:
        patches = [mpatches.Patch(color=cmap(i), label=celltype_names[t]) 
                   for i, t in enumerate(unique_types)]
        ax.legend(handles=patches, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved node attention scores to {save_path}")
    else:
        plt.show()


def create_attention_report(
    model,
    data,
    output_dir: str = 'attention_viz',
    node_indices: Optional[List[int]] = None,
    top_k_neighbors: int = 20,
    top_k_nodes: int = 50,
    exclude_celltypes: Optional[List[str]] = None
):
    """
    Create a comprehensive attention visualization report.
    
    Args:
        model: Trained GTVelo model
        data: Data dictionary
        output_dir: Directory to save visualizations
        node_indices: Specific nodes to visualize (if None, picks random ones)
        top_k_neighbors: Number of neighbors to show in subgraph
        top_k_nodes: Number of top nodes to show in attention scores
        exclude_celltypes: List of cell type names to exclude from neighbor visualization
                          (e.g., ['Primitive erythroid cells'])
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Extracting attention weights...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract attention weights
    attention_list = extract_attention_weights(
        model, data['X'], data['edge_index'], 
        data.get('edge_attr'), device=device
    )
    
    print(f"Extracted attention from {len(attention_list)} layers")
    
    # Aggregate attention
    print("Aggregating attention weights...")
    aggregated_attn = aggregate_attention_weights(attention_list, method='mean')
    
    # 1. Attention heatmap between cell types
    print("Creating attention heatmap...")
    visualize_attention_heatmap(
        aggregated_attn,
        data['edge_index'],
        celltype_names=data.get('celltype_names'),
        labels=data['labels'],
        save_path=str(output_path / 'attention_heatmap.png')
    )
    
    # 2. Top nodes by attention
    print("Visualizing top nodes by attention...")
    visualize_node_attention_scores(
        aggregated_attn,
        data['edge_index'],
        num_nodes=data['X'].shape[0],
        celltype_names=data.get('celltype_names'),
        labels=data['labels'],
        top_k=top_k_nodes,
        save_path=str(output_path / 'top_attention_nodes.png')
    )
    
    # 3. Subgraph visualizations for specific nodes
    if node_indices is None:
        # Pick some interesting nodes (e.g., from test set, different cell types)
        test_mask = data.get('test_mask', torch.zeros(data['X'].shape[0], dtype=torch.bool))
        if test_mask.sum() > 0:
            test_indices = torch.where(test_mask)[0].tolist()
            node_indices = test_indices[:5]  # First 5 test nodes
        else:
            # Random selection
            node_indices = np.random.choice(data['X'].shape[0], min(5, data['X'].shape[0]), replace=False).tolist()
    
    print(f"Creating subgraph visualizations for {len(node_indices)} nodes...")
    if exclude_celltypes:
        print(f"Excluding cell types: {exclude_celltypes}")
    for i, node_idx in enumerate(node_indices):
        visualize_attention_subgraph(
            data,
            aggregated_attn,
            node_idx=node_idx,
            top_k=top_k_neighbors,
            celltype_names=data.get('celltype_names'),
            timepoint=data.get('timepoint'),
            exclude_celltypes=exclude_celltypes,
            save_path=str(output_path / f'attention_subgraph_node_{node_idx}.png')
        )
    
    print(f"\nAttention visualization report saved to {output_path}/")
    print(f"- attention_heatmap.png: Attention between cell types")
    print(f"- top_attention_nodes.png: Top nodes by attention score")
    print(f"- attention_subgraph_node_*.png: Individual cell attention patterns")


if __name__ == "__main__":
    # Example usage
    print("Attention Visualization Module for GTVelo")
    print("Use create_attention_report() to generate visualizations from a trained model.")

