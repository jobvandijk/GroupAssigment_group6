from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    balanced_accuracy = per_class_recall.mean()
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
    plt.show()

print("Confusion Matrix:", cm_plot(y_true, y_pred, class_name='ERK2'))
print("Confusion Matrix:", cm_plot(y_true, y_pred, class_name="PKM2"))


# Confusion Matrix for ERK2 inhibition
cm_pkm2 = confusion_matrix(y_test[:, 0], y_pred[:, 0])
cm_erk2 = confusion_matrix(y_test[:, 1], y_pred[:, 1])
plt.subplot(1, 2, 2)
sns.heatmap(cm_erk2, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for ERK2 inhibition')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
