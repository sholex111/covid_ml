

#Run immediately after models have been built



# Load required libraries
library(caret)
library(randomForest)
library(rpart)
library(xgboost)

# Logistic Regression Model - Metrics
log_conf_matrix <- confusionMatrix(as.factor(log_pred_class), as.factor(testData$death), positive = "1")
log_sensitivity <- log_conf_matrix$byClass["Sensitivity"]
log_specificity <- log_conf_matrix$byClass["Specificity"]
log_npv <- log_conf_matrix$byClass["Neg Pred Value"]
log_ppv <- log_conf_matrix$byClass["Pos Pred Value"]

# Random Forest Model - Metrics
rf_conf_matrix <- confusionMatrix(rf_pred, as.factor(testData$death), positive = "1")
rf_sensitivity <- rf_conf_matrix$byClass["Sensitivity"]
rf_specificity <- rf_conf_matrix$byClass["Specificity"]
rf_npv <- rf_conf_matrix$byClass["Neg Pred Value"]
rf_ppv <- rf_conf_matrix$byClass["Pos Pred Value"]

# Decision Tree Model - Metrics
dt_conf_matrix <- confusionMatrix(dt_pred, as.factor(testData$death), positive = "1")
dt_sensitivity <- dt_conf_matrix$byClass["Sensitivity"]
dt_specificity <- dt_conf_matrix$byClass["Specificity"]
dt_npv <- dt_conf_matrix$byClass["Neg Pred Value"]
dt_ppv <- dt_conf_matrix$byClass["Pos Pred Value"]

# XGBoost Model - Metrics
xgb_conf_matrix <- confusionMatrix(as.factor(xgb_pred_class), as.factor(test_label), positive = "1")
xgb_sensitivity <- xgb_conf_matrix$byClass["Sensitivity"]
xgb_specificity <- xgb_conf_matrix$byClass["Specificity"]
xgb_npv <- xgb_conf_matrix$byClass["Neg Pred Value"]
xgb_ppv <- xgb_conf_matrix$byClass["Pos Pred Value"]
#### K-Nearest Neighbors (KNN) - Metrics ####

# Evaluate performance on the test data
knn_test_conf_matrix <- confusionMatrix(as.factor(knn_test_pred), as.factor(testData$death), positive = "1")
knn_sensitivity <- knn_test_conf_matrix$byClass["Sensitivity"]
knn_specificity <- knn_test_conf_matrix$byClass["Specificity"]
knn_npv <- knn_test_conf_matrix$byClass["Neg Pred Value"]
knn_ppv <- knn_test_conf_matrix$byClass["Pos Pred Value"]

# Train and Test Accuracy
knn_train_accuracy <- mean(knn_train_pred == trainData$death)
knn_test_accuracy <- mean(knn_test_pred == testData$death)

#### Support Vector Machine (SVM) - Metrics ####

# Evaluate performance on the test data
svm_test_conf_matrix <- confusionMatrix(as.factor(svm_test_pred), as.factor(testData$death), positive = "1")
svm_sensitivity <- svm_test_conf_matrix$byClass["Sensitivity"]
svm_specificity <- svm_test_conf_matrix$byClass["Specificity"]
svm_npv <- svm_test_conf_matrix$byClass["Neg Pred Value"]
svm_ppv <- svm_test_conf_matrix$byClass["Pos Pred Value"]

# Train and Test Accuracy
svm_train_accuracy <- mean(svm_train_pred == trainData$death)
svm_test_accuracy <- mean(svm_test_pred == testData$death)

#### Create Results Data Frame with KNN and SVM Metrics ####

results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "Decision Tree", "XGBoost", "K-Nearest Neighbors", "Support Vector Machine"),
  Train_Accuracy = c(log_train_accuracy, rf_train_accuracy, dt_train_accuracy, xgb_train_accuracy, knn_train_accuracy, svm_train_accuracy),
  Test_Accuracy = c(log_test_accuracy, rf_test_accuracy, dt_test_accuracy, xgb_test_accuracy, knn_test_accuracy, svm_test_accuracy),
  Sensitivity = c(log_sensitivity, rf_sensitivity, dt_sensitivity, xgb_sensitivity, knn_sensitivity, svm_sensitivity),
  Specificity = c(log_specificity, rf_specificity, dt_specificity, xgb_specificity, knn_specificity, svm_specificity),
  NPV = c(log_npv, rf_npv, dt_npv, xgb_npv, knn_npv, svm_npv),
  PPV = c(log_ppv, rf_ppv, dt_ppv, xgb_ppv, knn_ppv, svm_ppv)
)

# Display the results table
print(results)




