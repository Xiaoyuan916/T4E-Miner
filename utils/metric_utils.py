from sklearn.metrics import confusion_matrix

def print_detailed_table(truths, preds, epoch_tag=""):
    
    cm = confusion_matrix(truths, preds, labels=[0, 1])
    print("\n" + f"--- {epoch_tag} ---")
    header = f"{'Category':<12} | {'TP_Count':<10} | {'Pred_Total':<10} | {'Real_Total':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    pos_f1 = 0.0
    for cls in [0, 1]:
        p_really = cm[cls, cls]
        p_total = cm[:, cls].sum()
        gt_total = cm[cls, :].sum()
        precision = p_really / p_total if p_total > 0 else 0.0
        recall = p_really / gt_total if gt_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"Category_{cls:<3} | {p_really:<10} | {p_total:<10} | {gt_total:<10} | {precision:.4f}       | {recall:.4f}       | {f1:.4f}")
        if cls == 1: pos_f1 = f1
    print("=" * len(header) + "\n")
    return pos_f1