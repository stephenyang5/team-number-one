
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from GTVelo.model_transformer import create_model

"""
Purpose: Load trained GTVelo model and prepared data
Parameter: checkpoint path, data path
Return: model, data checkpoint
"""
def load_model_and_data(checkpoint_path, data_path):
    print("Loading model and data...")
    
    # Load data
    data = torch.load(data_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Create model
    model = create_model(
        num_classes=data['num_classes'],
        **checkpoint['config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return model, data, checkpoint

"""
Purpose: Evaluate GTVelo on test set
Parameter: model, data, device (if Oscar)
Return: Metric evaluation results
"""
def evaluate_gtvelo(model, data, device='cpu'):
    
    print("GTVELO EVALUATION")
    
    model = model.to(device)
    model.eval()


    # move to device
    X = data['X'].to(device)
    edge_index = data['edge_index'].to(device)
    edge_attr = data['edge_attr'].to(device)
    labels = data['labels'].numpy()
    test_mask = data['test_mask'].numpy()
    
    # Get predictions
    with torch.no_grad():
        logits, embeddings = model(X, edge_index, edge_attr)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    
    # Test set metrics
    test_labels = labels[test_mask]
    test_preds = preds[test_mask]
    test_probs = probs[test_mask]
    
    accuracy = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\nOverall Performance:")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tF1 (macro): {f1_macro:.4f}")
    print(f"\tF1 (weighted): {f1_weighted:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class Performance:")
    for i, name in enumerate(data['celltype_names']):
        class_mask = test_labels == i
        if class_mask.sum() > 0:
            class_acc = (test_preds[class_mask] == test_labels[class_mask]).mean()
            n_samples = class_mask.sum()
            print(f"\t{name:30} Acc: {class_acc:.4f} ({n_samples:4d} cells)")
    
    # store in a results dict
    results = { 
        'method': 'GTVelo',
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': test_preds,
        'probabilities': test_probs,
        'labels': test_labels,
        'embeddings': embeddings.cpu().numpy()[test_mask]
    }
    
    return results

"""
Purpose: Baseline Model ==> Log. Regression on PCA features
Params: Data
Return: Metric evaluation results
"""
def evaluate_logistic_regression(data):
    
    # Get data
    X = data['X'].numpy()
    labels = data['labels'].numpy()
    train_mask = data['train_mask'].numpy()
    test_mask = data['test_mask'].numpy()
    
    # Train logistic regression
    print("Training logistic regression baseline")
    clf = LogisticRegression(max_iter=1000, random_state=92)
    clf.fit(X[train_mask], labels[train_mask])
    
    # Predict
    test_preds = clf.predict(X[test_mask])
    test_probs = clf.predict_proba(X[test_mask])
    test_labels = labels[test_mask]
    
    # Metrics
    accuracy = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\nOverall Performance:")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tF1 (macro): {f1_macro:.4f}")
    print(f"\tF1 (weighted): {f1_weighted:.4f}")
    
    # results dict
    results = {
        'method': 'Logistic Regression',
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': test_preds,
        'probabilities': test_probs,
        'labels': test_labels
    }
    
    return results


"""
Purpose: KNN baseline (gold standard for single-cell)
Params: Data, n_neighbors
Return: Metric evaluation results
"""
def evaluate_knn(data, n_neighbors=15):
    
    print(f"\nEVALUATING KNN (k={n_neighbors})")
    print("-" * 50)
    
    # Get data
    X = data['X'].numpy()
    labels = data['labels'].numpy()
    train_mask = data['train_mask'].numpy()
    test_mask = data['test_mask'].numpy()
    
    # Train KNN
    print(f"Training KNN with {n_neighbors} neighbors...")
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(X[train_mask], labels[train_mask])
    
    # Predict
    test_preds = clf.predict(X[test_mask])
    test_probs = clf.predict_proba(X[test_mask])
    test_labels = labels[test_mask]
    
    # Metrics
    accuracy = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\nOverall Performance:")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tF1 (macro): {f1_macro:.4f}")
    print(f"\tF1 (weighted): {f1_weighted:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class Performance:")
    for i, name in enumerate(data['celltype_names']):
        class_mask = test_labels == i
        if class_mask.sum() > 0:
            class_acc = (test_preds[class_mask] == test_labels[class_mask]).mean()
            n_samples = class_mask.sum()
            print(f"\t{name:30} Acc: {class_acc:.4f} ({n_samples:4d} cells)")
    
    # Results dict
    results = {
        'method': f'KNN (k={n_neighbors})',
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': test_preds,
        'probabilities': test_probs,
        'labels': test_labels
    }
    
    return results


"""
Purpose: Random Forest baseline (robust non-linear classifier)
Params: Data, n_estimators, max_depth
Return: Metric evaluation results
"""
def evaluate_random_forest(data, n_estimators=100, max_depth=None):
    
    print(f"\nEVALUATING RANDOM FOREST (trees={n_estimators})")
    print("-" * 50)
    
    # Get data
    X = data['X'].numpy()
    labels = data['labels'].numpy()
    train_mask = data['train_mask'].numpy()
    test_mask = data['test_mask'].numpy()
    
    # Train Random Forest
    print(f"Training Random Forest with {n_estimators} trees...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=92,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(X[train_mask], labels[train_mask])
    
    # Predict
    test_preds = clf.predict(X[test_mask])
    test_probs = clf.predict_proba(X[test_mask])
    test_labels = labels[test_mask]
    
    # Metrics
    accuracy = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average='macro')
    f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    
    print(f"\nOverall Performance:")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tF1 (macro): {f1_macro:.4f}")
    print(f"\tF1 (weighted): {f1_weighted:.4f}")
    
    # Feature importance (top 10 PCs)
    importances = clf.feature_importances_
    top_features = np.argsort(importances)[::-1][:10]
    print(f"\nTop 10 Important Features (PCs):")
    for i, feat in enumerate(top_features, 1):
        print(f"\t{i}. PC{feat}: {importances[feat]:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class Performance:")
    for i, name in enumerate(data['celltype_names']):
        class_mask = test_labels == i
        if class_mask.sum() > 0:
            class_acc = (test_preds[class_mask] == test_labels[class_mask]).mean()
            n_samples = class_mask.sum()
            print(f"\t{name:30} Acc: {class_acc:.4f} ({n_samples:4d} cells)")
    
    # Results dict
    results = {
        'method': 'Random Forest',
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': test_preds,
        'probabilities': test_probs,
        'labels': test_labels,
        'feature_importances': importances
    }
    
    return results

"""
Purpose: Run all baseline methods
Params: Data
Return: List of all results
"""
def evaluate_all_baselines(data):
    
    print("\n" + "="*70)
    print("EVALUATING ALL BASELINE METHODS")
    print("="*70)
    
    results_list = []
    
    
    # KNN
    try:
        knn_results = evaluate_knn(data, n_neighbors=15)
        results_list.append(knn_results)
    except Exception as e:
        print(f"KNN failed: {e}")
        results_list.append(None)
    
    # Random Forest
    try:
        rf_results = evaluate_random_forest(data, n_estimators=100)
        results_list.append(rf_results)
    except Exception as e:
        print(f"Random Forest failed: {e}")
        results_list.append(None)
    
    return results_list



"""
Purpose: Confusion Matrix
Params: results, celltype labels
Return: None
"""
def plot_confusion_matrix(results, celltype_names, save_path=None):

    cm = confusion_matrix(results['labels'], results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=celltype_names,
                yticklabels=celltype_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{results["method"]} Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # tight is messy with mouse data
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()

"""
Purpose: Compare all the methods and plot
Params: results list
Return: dataframe w/ comparisons
"""
def compare_methods(results_list, save_path=None):
    
    print("METHOD COMPARISON")
    
    # Create comparison table
    comparison = []
    for result in results_list:
        if result is not None:
            comparison.append({
                'Method': result['method'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'F1 (macro)': f"{result['f1_macro']:.4f}",
                'F1 (weighted)': f"{result['f1_weighted']:.4f}"
            })
    
    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n Saved comparison to {save_path}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [r['method'] for r in results_list if r is not None]
    accuracies = [r['accuracy'] for r in results_list if r is not None]
    f1_macros = [r['f1_macro'] for r in results_list if r is not None]
    f1_weighteds = [r['f1_weighted'] for r in results_list if r is not None]
    
    # Accuracy
    axes[0].bar(methods, accuracies, color='skyblue')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # F1 macro
    axes[1].bar(methods, f1_macros, color='lightcoral')
    axes[1].set_ylabel('F1 (macro)')
    axes[1].set_title('F1 Score (macro)')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1 weighted
    axes[2].bar(methods, f1_weighteds, color='magenta')
    axes[2].set_ylabel('F1 (weighted)')
    axes[2].set_title('F1 Score (weighted)')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plot_path = str(save_path).replace('.csv', '_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {plot_path}")
    else:
        plt.show()
    
    plt.close()
    
    return df

    """
Purpose: Plot feature importance from Random Forest
Params: RF results, save path
Return: None
"""
def plot_feature_importance(rf_results, n_features=20, save_path=None):
    
    if 'feature_importances' not in rf_results:
        print("No feature importances found in results")
        return
    
    importances = rf_results['feature_importances']
    indices = np.argsort(importances)[::-1][:n_features]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_features), importances[indices])
    plt.xlabel('Principal Component')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importance (Top PCs)')
    plt.xticks(range(n_features), [f'PC{i}' for i in indices], rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


"""Main script"""
def main():
    # # Paths (manually change for different datasets!!!)
    # checkpoint_path = 'checkpoints/gtvelo_best.pt'
    # data_path = 'mouse_ethyroid/mouse_ethyroid_prepared_swap.pt'
    # adata_path = 'mouse_ethyroid/mouse_ethyroid_data.h5ad'
    # results_dir = Path('results')
    # results_dir.mkdir(exist_ok=True)
    
    # # Load model and data
    # model, data, checkpoint = load_model_and_data(checkpoint_path, data_path)
    
    # # Evaluate GTVelo
    # print("\n" + "-"*50)
    # print("EVALUATING GTVELO")
    # gtvelo_results = evaluate_gtvelo(model, data)
    
    # # Evaluate all baselines
    # print("\n" + "-"*50)
    # print("EVALUATING BASELINE METHODS")
    
    # lr_results = evaluate_logistic_regression(data)
    # knn_results = evaluate_knn(data, n_neighbors=15)
    # rf_results = evaluate_random_forest(data, n_estimators=100)
    
    # # Collect all results
    # results_list = [gtvelo_results, lr_results, knn_results, rf_results]
    
    # # Plot confusion matrices for all methods
    # for result in results_list:
    #     if result is not None:
    #         method_name = result['method'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    #         plot_confusion_matrix(
    #             result,
    #             data['celltype_names'],
    #             save_path=results_dir / f'{method_name}_confusion_matrix.png'
    #         )
    
    # # Plot Random Forest feature importance
    # if rf_results is not None:
    #     plot_feature_importance(
    #         rf_results,
    #         n_features=20,
    #         save_path=results_dir / 'random_forest_feature_importance.png'
    #     )
    
    # # Compare all methods
    # comparison_df = compare_methods(
    #     results_list,
    #     save_path=results_dir / 'method_comparison.csv'
    # )
    
    # # Print classification reports
    # print("\n" + "="*70)
    # print("CLASSIFICATION REPORTS")
    
    # for result in results_list:
    #     if result is not None:
    #         print(f"\n{result['method']}:")
    #         print(classification_report(
    #             result['labels'],
    #             result['predictions'],
    #             target_names=data['celltype_names'],
    #             digits=4
    #         ))
    
    # print("\n" + "="*70)
    # print("EVALUATION COMPLETE!")
    # print("="*70)
    # print(f"\nResults saved to: {results_dir}/")
    
    # # Summary comparison
    # print(f"\nSUMMARY:")
    # print(f"\t{'Method':<25} {'Accuracy':<10} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
    # print(f"\t{'-'*60}")
    # for result in results_list:
    #     if result is not None:
    #         print(f"\t{result['method']:<25} {result['accuracy']:<10.4f} {result['f1_macro']:<12.4f} {result['f1_weighted']:<12.4f}")
    
    # # Best method
    # best_method = max(results_list, key=lambda x: x['accuracy'] if x else 0)
    # print(f"\n\t🏆 Best method: {best_method['method']} ({best_method['accuracy']:.4f})")
    print("running evaluate.py")
if __name__ == "__main__":
    main()