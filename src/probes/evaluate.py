import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


def evaluate_classification(y_true, y_pred_probs, threshold=None):
    
    if threshold is None:
        threshold = np.percentile(y_pred_probs, 95)
        print(f"Automatically selected threshold: {threshold:.3f}")
        
    y_pred_labels = (y_pred_probs >= threshold).astype(int)
    
    auprc = average_precision_score(y_true, y_pred_probs)
    auroc = roc_auc_score(y_true, y_pred_probs)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    
    # TPR and FPR at threshold
    tn = ((y_true == 0) & (y_pred_labels == 0)).sum()
    fp = ((y_true == 0) & (y_pred_labels == 1)).sum()
    fn = ((y_true == 1) & (y_pred_labels == 0)).sum()
    tp = ((y_true == 1) & (y_pred_labels == 1)).sum()
    
    tpr_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Precision@100
    sorted_indices = np.argsort(-y_pred_probs)
    top_100_indices = sorted_indices[:100]
    precision_at_100 = precision_score(y_true[top_100_indices], y_pred_labels[top_100_indices])
    
    # Enrichment Factor @1% and @5%
    n = len(y_true)
    top_1_percent = max(1, int(np.ceil(0.01 * n)))
    top_5_percent = max(1, int(np.ceil(0.05 * n)))
    
    ef1 = (y_true[sorted_indices[:top_1_percent]].sum() / top_1_percent) / (y_true.sum() / n)
    ef5 = (y_true[sorted_indices[:top_5_percent]].sum() / top_5_percent) / (y_true.sum() / n)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_labels)
    
    # F1 Score
    f1 = f1_score(y_true, y_pred_labels)
    
    return {
        'AUPRC': auprc,
        'AUROC': auroc,
        'Precision': precision,
        'Recall': recall,
        'Precision@100': precision_at_100,
        'EF@1%': ef1,
        'EF@5%': ef5,
        'Accuracy': accuracy,
        'FPR@threshold': fpr_at_threshold,
        'TPR@threshold': tpr_at_threshold,
        'F1': f1
    }
    
    
def display_results(results):
    for key, value in results.items():
        print(f"{key}: {value:.3f}")
    return None        


def plot_pr_curve(y_true, y_pred_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()
    return None
