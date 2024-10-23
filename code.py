# Importing libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Reading the dataset and removing empty columns
DATA_PATH = "Training.csv"
data = pd.read_csv('C:\\Users\\abc\\Downloads\\ML\\pythonProject\\Dataset---Disease-Prediction-Using--Machine-Learning.csv').dropna(axis=1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.title("Disease Distribution")
plt.xticks(rotation=45)
plt.show()

# Encoding the target value into numerical values
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# Defining scoring metric for cross-validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


# Initializing Models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Check the distribution of classes
class_counts = data["prognosis"].value_counts()
print("Class distribution:\n", class_counts)

# Set the number of splits based on the smallest class size, with a minimum of 2
n_splits = max(2, min(5, class_counts.min()))  # Ensures at least 2 splits

# Use Stratified K-Fold with the determined number of splits
skf = StratifiedKFold(n_splits=n_splits)

# Producing cross-validation scores for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=skf, n_jobs=-1, scoring=cv_scoring)
    print("==" * 30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
print(f"Accuracy on train data by SVM Classifier: {accuracy_score(y_train, svm_model.predict(X_train)) * 100:.2f}%")
print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, svm_preds) * 100:.2f}%")

cf_matrix = confusion_matrix(y_test, svm_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
print(
    f"Accuracy on train data by Naive Bayes Classifier: {accuracy_score(y_train, nb_model.predict(X_train)) * 100:.2f}%")
print(f"Accuracy on test data by Naive Bayes Classifier: {accuracy_score(y_test, nb_preds) * 100:.2f}%")

cf_matrix = confusion_matrix(y_test, nb_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(
    f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train)) * 100:.2f}%")
print(f"Accuracy on test data by Random Forest Classifier: {accuracy_score(y_test, rf_preds) * 100:.2f}%")

cf_matrix = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Training the models on the whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Making predictions by taking the mode of predictions from all classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [stats.mode([i, j, k]).mode[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds) * 100:.2f}%")

cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Creating a symptom index dictionary to encode the input symptoms into numerical form
symptoms = X.columns.values
symptom_index = {symptom: index for index, symptom in enumerate(symptoms)}
data_dict = {"symptom_index": symptom_index, "predictions_classes": encoder.classes_}


# Defining the prediction function
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    predictions = {}

    for symptom in symptoms:
        symptom = symptom.strip().capitalize()  # Clean input
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            print(f"Warning: '{symptom}' is not recognized.")

    input_data = np.array(input_data).reshape(1, -1)

    # Generate predictions
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Final prediction using mode
    final_prediction = stats.mode([rf_prediction, nb_prediction, svm_prediction]).mode[0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions


# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
