

#####
## Hyperparameter tunning

# Required Libraries
library(caret)
library(glmnet)
library(xgboost)
library(dplyr)


trainData$death <- as.factor(trainData$death)
testData$death <- as.factor(testData$death)

# Rename levels of the target variable to be valid R variable names
levels(trainData$death) <- c("No", "Yes")
levels(testData$death) <- c("No", "Yes")

# Hyperparameter grids for each model
# Logistic Regression - using glmnet for regularization
log_grid <- expand.grid(
  alpha = c(0, 1),        # Lasso (1) and Ridge (0)
  lambda = 10^seq(-3, 1, length = 10)
)


# Simplified Random Forest Grid
rf_grid <- expand.grid(mtry = c(2, 3, 4))


# XGBoost Grid
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 5),
  eta = c(0.01, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = c(1, 3),
  subsample = c(0.7, 1)
)

# Define trainControl
train_control <- trainControl(
  method = "cv", number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final"
)


# Logistic Regression Model
log_model <- train(
  death ~ ., data = trainData,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = log_grid,
  family = "binomial",
  metric = "ROC"
)



# Random Forest Model
rf_model <- train(
  death ~ ., data = trainData,
  method = "rf",
  trControl = train_control,
  tuneGrid = rf_grid,
  metric = "ROC"
)



# Run the XGBoost model with early stopping
xgb_model <- train(
  death ~ ., data = trainData,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "ROC",
  # Set early stopping rounds to stop if no improvement over 10 rounds
  tuneLength = 10  # indirectly control depth of tuning if not included directly
)


# Store the top 5 results for each model on train data
top_5_results <- function(model) {
  res <- model$results %>%
    arrange(desc(ROC)) %>%
    slice(1:5) %>%
    select(ROC, Sens, Spec)
  return(res)
}

log_results <- top_5_results(log_model)
rf_results <- top_5_results(rf_model)
xgb_results <- top_5_results(xgb_model)



# Logistic Regression Performance on Test Data
log_test_pred <- predict(log_model, newdata = testData, type = "raw")
log_test_perf <- confusionMatrix(log_test_pred, testData$death, positive = "Yes")

# Random Forest Performance on Test Data
rf_test_pred <- predict(rf_model, newdata = testData, type = "raw")
rf_test_perf <- confusionMatrix(rf_test_pred, testData$death, positive = "Yes")

# XGBoost Performance on Test Data
xgb_test_pred <- predict(xgb_model, newdata = testData, type = "raw")
xgb_test_perf <- confusionMatrix(xgb_test_pred, testData$death, positive = "Yes")

# Print results
log_test_perf
rf_test_perf
xgb_test_perf




# Get training performance metrics for each model
log_results <- getTrainPerf(log_model)
rf_results <- getTrainPerf(rf_model)
xgb_results <- getTrainPerf(xgb_model)

# Combine results into a final table
tuning_table <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Train_ROC = c(log_results$TrainROC, rf_results$TrainROC, xgb_results$TrainROC),
  Train_Sensitivity = c(log_results$TrainSens, rf_results$TrainSens, xgb_results$TrainSens),
  Train_Specificity = c(log_results$TrainSpec, rf_results$TrainSpec, xgb_results$TrainSpec),
  Test_Accuracy = c(log_test_perf$overall['Accuracy'], rf_test_perf$overall['Accuracy'], xgb_test_perf$overall['Accuracy']),
  Test_Sensitivity = c(log_test_perf$byClass['Sensitivity'], rf_test_perf$byClass['Sensitivity'], xgb_test_perf$byClass['Sensitivity']),
  Test_Specificity = c(log_test_perf$byClass['Specificity'], rf_test_perf$byClass['Specificity'], xgb_test_perf$byClass['Specificity'])
)

# Display the final table
print(tuning_table)


