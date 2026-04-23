# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    """
    Generates and saves a high-quality heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    
    # Using a clean color palette (Blues or Greens) for academic look
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Behavioral Classification Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300) # Save with high resolution for paper
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_classification_report(y_true, y_pred, classes, save_path="classification_report.csv"):
    """
    Generates a detailed F1, Precision, and Recall report and saves as CSV.
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path)
    print(f"Detailed classification report saved to {save_path}")