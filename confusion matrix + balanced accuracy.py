from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    sensitivities = []

    for i in range(num_classes):
        true_positives = cm[i, i]
        actual_positives = cm[i, :].sum()
        sensitivity = true_positives / actual_positives
        sensitivities.append(sensitivity)

    balanced_accuracy = sum(sensitivities) / num_classes
    return balanced_accuracy

# Example usage
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # True labels
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]  # Predicted labels

print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))


def cm_plot(y_true, y_pred, class_name):
    # Confusion Matrix for PKM2 inhibition
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix for "+ class_name + " inhibition")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


print("Confusion Matrix:", cm_plot(y_true, y_pred, class_name='ERK2'))
print("Confusion Matrix:", cm_plot(y_true, y_pred, class_name="PKM2"))



