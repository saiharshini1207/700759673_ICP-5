import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
glass_df = pd.read_csv('C:\\Users\\Harshini\\Documents\\NN python\\glass.csv')
# Split dataset into features and target variable
X = glass_df.drop('Type', axis=1)  # Features
y = glass_df['Type']  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the SVM model
svm_model = SVC(kernel='linear')

# Train the model using the training sets
svm_model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = svm_model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
accuracy = svm_model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Generate classification report
print(classification_report(y_test, y_pred))

print("Linear SVM Accuracy:", accuracy)
