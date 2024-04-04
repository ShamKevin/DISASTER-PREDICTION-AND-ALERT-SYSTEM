import os
import pandas as pd
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Import joblib for saving the model

data_dir = "data" # Update with your data directory
test_size = 0.1  

subfolders = os.listdir(data_dir)

data = []
for cls in subfolders:
    cls_dir = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img_name)
        data.append((img_path, cls))

train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

train_df = pd.DataFrame(train_data, columns=["image_path", "class"])
test_df = pd.DataFrame(test_data, columns=["image_path", "class"])

# Define image preprocessing function
def preprocess_image(img_path):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        new_array = cv2.resize(img_array, (50, 50))  # Adjust the size as needed
        return new_array.flatten()
    except Exception as e:
        return None

# Create training data
training_data = []

for index, row in train_df.iterrows():
    img_path = row["image_path"]
    class_num = subfolders.index(row["class"])
    img_data = preprocess_image(img_path)
    if img_data is not None:
        training_data.append([img_data, class_num])

random.shuffle(training_data)

X = []  # Features
y = []  # Labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X*3)
y = np.array(y*3)

print("Image preprocessing completed.")

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Building the SVM model
svm_model = SVC(kernel='linear')  # You can experiment with different kernels (linear, radial basis function, etc.)

# Training the SVM model
svm_model.fit(X_train, y_train)

# Save the trained SVM model as a pickle file
joblib.dump(svm_model, 'SVM_model.pkl')

# Making predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Print classification report and confusion matrix
print('\nClassification Report\n')
print(classification_report(y_test, y_pred))

confusion_mtx = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap="Blues", linewidths=0.5, cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

