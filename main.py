from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

print(f"The breast cancer data features:\n{breast_cancer_data.feature_names}\n")

print(f"The breast cancer data target:\n{breast_cancer_data.target}\n")

print(f"The breast cancer data target names:\n{breast_cancer_data.target_names}\n")

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(f"The length of the training data:\n{len(training_data)}\n")
print(f"The length of the training labels:\n{len(training_labels)}\n")

k_list = range(1,101)
accuracies = []
max_accuracy = 0
max_accuracy_k = 0

for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    accuracy = classifier.score(validation_data, validation_labels)
    if accuracy >= max_accuracy:
        max_accuracy = accuracy
        max_accuracy_k = k
    accuracies.append(accuracy)

print(f"For k = {max_accuracy_k}, the classifier's accuracy scored a maximum accuracy of {max_accuracy}")

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()