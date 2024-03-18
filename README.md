# Naive Bayes on Credit Card Dataset
please see [report.pdf](https://github.com/cjohns26/Naive-Bayes_Credit-Card-Dataset/blob/main/Naive-Bayes-Algo/report.pdf) for full description.

The algorithm assumes that the probabilities of features are independent of each other. The experiment is based on training the model on 10,000 records of credit card data, focusing on five features: Geography, isActiveMember, HasCrCard, Balance, and CreditScore. The goal is to predict the class (Exited) based on these features. The model’s accuracy is then tested on 2,000 records, resulting in an accuracy of 80.50%. However, further analysis reveals that the training data heavily favors the 0 class, which explains the model’s tendency to predict 0 consistently. Future work could involve manipulating the test data to observe if the model ever predicts the 1 class. Overall, the Naive Bayes algorithm shows promise in predicting the class but may require further refinement and evaluation with balanced data.



