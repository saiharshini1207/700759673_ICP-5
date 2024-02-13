CODE EXPLANATION LINK:


# NAIVE BAYES METHOD:

1. This Python code loads the 'glass.csv' dataset using pandas, splits it into features (X) and the target variable (y).
2.  then further splits it into training and testing sets using the train_test_split function from scikit-learn. It then instantiates a Naïve Bayes model (specifically Gaussian Naïve Bayes), trains the model on the training data, and makes predictions on the test data.
3.   After that, it calculates the accuracy of the model on the test set using the score method and prints it.
4.    generates a classification report using the classification_report function from scikit-learn, which provides metrics such as precision, recall, and F1-score for each class.
5.Finally, it prints the accuracy.


# SVM METHOD

1. This Python code loads the 'glass.csv' dataset using pandas, splits it into features (X) and the target variable (y), and then further splits it into training and testing sets using the train_test_split function from scikit-learn.
2. It then instantiates a Support Vector Machine (SVM) model with a linear kernel, trains the model on the training data, and makes predictions on the test data.
3. After that, it calculates the accuracy of the model on the test set using the score method and prints.
4. it generates a classification report using the classification_report function from scikit-learn, which provides metrics such as precision, recall, and F1-score for each class. Finally, it prints the accuracy again to ensure visibility.


#COMPARISON BETWEEN NAIVE BAYES AND SVM METHOD:

The SVM achieved better accuracy with an accuracy score of 0.6769 compared to the Naïve Bayes classifier, which had an accuracy of 0.3077.


Data Separability: SVM is particularly effective when classes are well-separated, and there exists a clear margin of separation between them.

Model Complexity: Linear SVM is a more complex model compared to Naïve Bayes. It can capture nonlinear relationships between features through the use of kernels, even in its linear form. 

Handling of Features: Naïve Bayes might not capture these dependencies accurately. SVM, on the other hand, does not make such assumptions and can handle feature interactions more effectively.

Robustness to Outliers: SVM is generally more robust to outliers compared to Naïve Bayes.
