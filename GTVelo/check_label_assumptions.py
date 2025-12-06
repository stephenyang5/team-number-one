"""
Diagnostic script to check if labels are current state vs future fate
"""

import torch
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_label_assumptions(data_path='data/blood/blood_prepared.pt'):
    """
    Analyze whether labels represent current state or future fate.
    """
    print("="*70)
    print("ANALYZING LABEL ASSUMPTIONS")
    print("="*70)
    
    data = torch.load(data_path, map_location='cpu')
    
    labels = data['labels']
    timepoint = data['timepoint']
    celltype_names = data['celltype_names']
    train_mask = data['train_mask']
    test_mask = data['test_mask']
    
    print(f"\nCell types: {celltype_names}")
    print(f"Temporal cutoff: E{data['temporal_cutoff']}")
    
    # Analyze label distribution by timepoint
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION BY TIMEPOINT")
    print("="*70)
    
    unique_timepoints = sorted(timepoint.unique().tolist())
    
    for t in unique_timepoints:
        mask = timepoint == t
        print(f"\nE{t}: {mask.sum():,} cells")
        for i, name in enumerate(celltype_names):
            count = ((labels == i) & mask).sum().item()
            pct = 100 * count / mask.sum().item() if mask.sum() > 0 else 0
            if count > 0:
                print(f"  {name:35} {count:6,} ({pct:5.1f}%)")
    
    # Check if labels change over time (suggesting current state, not future fate)
    print("\n" + "="*70)
    print("CHECKING IF LABELS REPRESENT CURRENT STATE VS FUTURE FATE")
    print("="*70)
    
    print("\nIf labels are CURRENT STATE:")
    print("  - Early timepoints should have more 'progenitor' labels")
    print("  - Late timepoints should have more 'differentiated' labels")
    print("  - Same cell type should appear at multiple timepoints")
    
    print("\nIf labels are FUTURE FATE:")
    print("  - Early timepoints should have labels matching their future fate")
    print("  - Progenitor cells at E8.5 should be labeled as their E13.5 fate")
    print("  - Labels should be consistent across timepoints for same trajectory")
    
    # Analyze transitions
    print("\n" + "="*70)
    print("ANALYZING LABEL CONSISTENCY")
    print("="*70)
    
    # For each cell type, check its distribution across timepoints
    for i, name in enumerate(celltype_names):
        type_mask = labels == i
        if type_mask.sum() == 0:
            continue
            
        type_timepoints = timepoint[type_mask]
        timepoint_counts = {}
        for t in unique_timepoints:
            count = (type_timepoints == t).sum().item()
            if count > 0:
                timepoint_counts[t] = count
        
        if len(timepoint_counts) > 1:
            print(f"\n{name}:")
            print(f"  Appears at {len(timepoint_counts)} timepoints:")
            for t, count in sorted(timepoint_counts.items()):
                pct = 100 * count / type_mask.sum().item()
                print(f"    E{t}: {count:,} cells ({pct:.1f}% of this type)")
        else:
            t = list(timepoint_counts.keys())[0]
            count = timepoint_counts[t]
            print(f"\n{name}:")
            print(f"  ONLY appears at E{t}: {count:,} cells")
    
    # Check test set specifically
    print("\n" + "="*70)
    print("TEST SET ANALYSIS (E < cutoff)")
    print("="*70)
    print(f"Test cells: {test_mask.sum():,}")
    print("\nTest set label distribution:")
    for i, name in enumerate(celltype_names):
        count = ((labels == i) & test_mask).sum().item()
        pct = 100 * count / test_mask.sum().item() if test_mask.sum() > 0 else 0
        if count > 0:
            print(f"  {name:35} {count:6,} ({pct:5.1f}%)")
    
    # Check train set
    print("\n" + "="*70)
    print("TRAIN SET ANALYSIS (E >= cutoff)")
    print("="*70)
    print(f"Train cells: {train_mask.sum():,}")
    print("\nTrain set label distribution:")
    for i, name in enumerate(celltype_names):
        count = ((labels == i) & train_mask).sum().item()
        pct = 100 * count / train_mask.sum().item() if train_mask.sum() > 0 else 0
        if count > 0:
            print(f"  {name:35} {count:6,} ({pct:5.1f}%)")
    
    # Critical check: Are there progenitor cells in test set?
    print("\n" + "="*70)
    print("CRITICAL CHECK: PROGENITOR CELLS IN TEST SET")
    print("="*70)
    
    progenitor_types = [i for i, name in enumerate(celltype_names) 
                       if 'progenitor' in name.lower() or 'progenitor' in name]
    
    if progenitor_types:
        print(f"\nProgenitor cell types found: {[celltype_names[i] for i in progenitor_types]}")
        for i in progenitor_types:
            test_progenitors = ((labels == i) & test_mask).sum().item()
            train_progenitors = ((labels == i) & train_mask).sum().item()
            print(f"\n  {celltype_names[i]}:")
            print(f"    Test set (E<{data['temporal_cutoff']}): {test_progenitors:,} cells")
            print(f"    Train set (E>={data['temporal_cutoff']}): {train_progenitors:,} cells")
            
            if test_progenitors > 0 and train_progenitors == 0:
                print(f"    ⚠️  WARNING: Progenitors only in test set - labels likely CURRENT STATE")
            elif test_progenitors > 0 and train_progenitors > 0:
                print(f"    ✓ Progenitors in both sets - could be current state or persistent type")
    else:
        print("\nNo explicit 'progenitor' cell types found in labels")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nIf test set has mostly differentiated cell types (e.g., 'Primitive erythroid'),")
    print("then labels are likely CURRENT STATE annotations, not future fates.")
    print("\nFor true fate prediction, you would need:")
    print("  1. Lineage tracing data (experimental)")
    print("  2. Trajectory inference (RNA velocity, pseudotime)")
    print("  3. Or labels assigned based on downstream analysis")
    print("\nCurrent setup may be learning: 'Given current features, what is the current cell type?'")
    print("Rather than: 'Given progenitor features, what will be the future cell type?'")


if __name__ == "__main__":
    analyze_label_assumptions()

