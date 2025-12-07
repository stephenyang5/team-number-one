import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import json

"""
Purpose: Save per-class performance to CSV
Params: results, celltype_names, save_path
Return: DataFrame with per-class metrics
"""
def save_per_class_metrics(results, celltype_names, save_path=None):
    """
    Export detailed per-class performance metrics to CSV
    """
    test_labels = results['labels']
    test_preds = results['predictions']
    
    per_class_data = []
    
    for i, name in enumerate(celltype_names):
        class_mask = test_labels == i
        
        if class_mask.sum() > 0:
            n_samples = class_mask.sum()
            n_correct = (test_preds[class_mask] == test_labels[class_mask]).sum()
            accuracy = n_correct / n_samples
            
            # Calculate precision, recall, F1 for this class
            tp = ((test_preds == i) & (test_labels == i)).sum()
            fp = ((test_preds == i) & (test_labels != i)).sum()
            fn = ((test_preds != i) & (test_labels == i)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_data.append({
                'Cell Type': name,
                'N_Samples': int(n_samples),
                'N_Correct': int(n_correct),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })
        else:
            per_class_data.append({
                'Cell Type': name,
                'N_Samples': 0,
                'N_Correct': 0,
                'Accuracy': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1_Score': 0.0
            })
    
    df = pd.DataFrame(per_class_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved per-class metrics to {save_path}")
    
    return df


"""
Purpose: Save confusion matrix to CSV
Params: results, celltype_names, save_path
Return: DataFrame with confusion matrix
"""
def save_confusion_matrix_csv(results, celltype_names, save_path=None):
    """
    Export confusion matrix as CSV with row/column labels
    """
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # Create DataFrame with proper labels
    df = pd.DataFrame(cm, 
                     index=[f"True_{name}" for name in celltype_names],
                     columns=[f"Pred_{name}" for name in celltype_names])
    
    if save_path:
        df.to_csv(save_path)
        print(f"Saved confusion matrix to {save_path}")
    
    return df


"""
Purpose: Save full classification report to text file
Params: results, celltype_names, save_path
Return: None
"""
def save_classification_report(results, celltype_names, save_path=None):
    """
    Export sklearn classification report to text file
    """
    report = classification_report(
        results['labels'],
        results['predictions'],
        target_names=celltype_names,
        digits=4
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"Classification Report: {results['method']}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        print(f"Saved classification report to {save_path}")
    
    return report


"""
Purpose: Save comprehensive results for one method
Params: result, celltype_names, method_dir
Return: None
"""
def save_method_results(result, celltype_names, method_dir):
    """
    Save all results for a single method to a dedicated folder
    """
    # Create method-specific directory
    method_dir = Path(method_dir)
    method_dir.mkdir(exist_ok=True, parents=True)
    
    method_name = result['method'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    
    print(f"\nSaving results for {result['method']}...")
    
    # 1. Per-class metrics
    per_class_df = save_per_class_metrics(
        result, 
        celltype_names,
        save_path=method_dir / f'{method_name}_per_class_metrics.csv'
    )
    
    # 2. Confusion matrix
    cm_df = save_confusion_matrix_csv(
        result,
        celltype_names,
        save_path=method_dir / f'{method_name}_confusion_matrix.csv'
    )
    
    # 3. Classification report
    save_classification_report(
        result,
        celltype_names,
        save_path=method_dir / f'{method_name}_classification_report.txt'
    )
    
    # 4. Summary stats
    summary = {
        'method': result['method'],
        'accuracy': float(result['accuracy']),
        'f1_macro': float(result['f1_macro']),
        'f1_weighted': float(result['f1_weighted']),
        'n_samples': int(len(result['labels'])),
        'n_classes': len(celltype_names)
    }
    
    with open(method_dir / f'{method_name}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ All results saved to {method_dir}/")


"""
Purpose: Create comprehensive results summary
Params: results_list, celltype_names, results_dir
Return: None
"""
def save_all_results(results_list, celltype_names, results_dir):
    """
    Save comprehensive results for all methods
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING COMPREHENSIVE RESULTS")
    print("="*70)
    
    # Save results for each method
    for result in results_list:
        if result is not None:
            method_name = result['method'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            method_dir = results_dir / method_name
            save_method_results(result, celltype_names, method_dir)
    
    # Create overall comparison file with per-class breakdown
    all_per_class = []
    
    for result in results_list:
        if result is not None:
            per_class_df = save_per_class_metrics(result, celltype_names)
            per_class_df['Method'] = result['method']
            all_per_class.append(per_class_df)
    
    if all_per_class:
        combined_df = pd.concat(all_per_class, ignore_index=True)
        # Reorder columns
        cols = ['Method', 'Cell Type', 'N_Samples', 'N_Correct', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
        combined_df = combined_df[cols]
        combined_df.to_csv(results_dir / 'all_methods_per_class_comparison.csv', index=False)
        print(f"\n✓ Saved combined per-class comparison: all_methods_per_class_comparison.csv")
    
    # Create pivot table for easy comparison
    pivot_accuracy = combined_df.pivot(index='Cell Type', columns='Method', values='Accuracy')
    pivot_accuracy.to_csv(results_dir / 'per_class_accuracy_pivot.csv')
    print(f"✓ Saved accuracy pivot table: per_class_accuracy_pivot.csv")
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {results_dir}/")
    print(f"{'='*70}\n")


"""
Purpose: Main script that properly saves all the results in the folder
"""
def main():
    from evaluate import (load_model_and_data, evaluate_gtvelo, 
                         evaluate_logistic_regression, evaluate_knn, 
                         evaluate_random_forest, plot_confusion_matrix, 
                         compare_methods, plot_feature_importance)
    
    # Paths
    checkpoint_path = 'checkpoints/gtvelo_best.pt'
    data_path = 'neuron/neuron_prepared_velocity.pt'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print("-"*50)
    print("COMPREHENSIVE EVALUATION AND RESULTS EXPORT")
    
    # Load model and data
    model, data, checkpoint = load_model_and_data(checkpoint_path, data_path)
    
    # Evaluate GTVelo
    print("\n" + "-"*50)
    print("EVALUATING GTVELO")
    gtvelo_results = evaluate_gtvelo(model, data)
    
    # Evaluate baselines
    print("\n" + "-"*50)
    print("EVALUATING BASELINE METHODS")
    
    lr_results = evaluate_logistic_regression(data)
    knn_results = evaluate_knn(data, n_neighbors=15)
    rf_results = evaluate_random_forest(data, n_estimators=100)
    
    # Collect all results
    results_list = [gtvelo_results, lr_results, knn_results, rf_results]
    
    # Plot confusion matrices
    print("\n" + "-"*50)
    print("GENERATING CONFUSION MATRIX PLOTS")
    for result in results_list:
        if result is not None:
            method_name = result['method'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            plot_confusion_matrix(
                result,
                data['celltype_names'],
                save_path=results_dir / f'{method_name}_confusion_matrix.png'
            )
    
    # Plot RF feature importance
    if rf_results is not None:
        plot_feature_importance(
            rf_results,
            n_features=20,
            save_path=results_dir / 'random_forest_feature_importance.png'
        )
    
    # Overall comparison
    print("\n" + "-"*50)
    print("GENERATING COMPARISON CHARTS")
    comparison_df = compare_methods(
        results_list,
        save_path=results_dir / 'method_comparison.csv'
    )
    
    # Save comprehensive results
    save_all_results(results_list, data['celltype_names'], results_dir)
    
    # Print summary to console
    print("\n" + "="*70)
    print("SUMMARY")
    print(f"\n{'Method':<25} {'Accuracy':<10} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
    print("-"*60)
    for result in results_list:
        if result is not None:
            print(f"{result['method']:<25} {result['accuracy']:<10.4f} "
                  f"{result['f1_macro']:<12.4f} {result['f1_weighted']:<12.4f}")
    
    # Best method
    best_method = max(results_list, key=lambda x: x['accuracy'] if x else 0)
    print(f"\n🏆 Best method: {best_method['method']} ({best_method['accuracy']:.4f})")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {results_dir}/")
    print(f"\nGenerated files:")
    print(f"\t- Individual method folders with detailed metrics")
    print(f"\t- all_methods_per_class_comparison.csv (comprehensive)")
    print(f"\t- per_class_accuracy_pivot.csv (easy comparison)")
    print(f"\t- method_comparison.csv (overall metrics)")
    print(f"\t- Confusion matrices (PNG and CSV)")
    print(f"\t- Classification reports (TXT)")


if __name__ == "__main__":
    main()