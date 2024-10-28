

##Covid Death Prediction  #Machine learning with R

# # List of packages
# packages <- c("tidyverse","caret", "randomForest", "rpart", "xgboost","MASS","ROCR",
#               "corrplot")
# 
# # Install the packages
# install.packages(packages)


# Load necessary libraries
library(caret)
library(randomForest)
library(rpart)
library(xgboost)  # For XGBoost
library(MASS)  # For Logistic Regression
library(ROCR)  # For ROC and AUC
library(dplyr)
library(ggplot2)
library(corrplot)
library(readxl)
library(janitor)



#load data
raw_data <- read_excel("Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.xlsx")

#add ID
raw_data <- raw_data %>% mutate(id = row_number()) %>%   select(id, everything())

raw_data <- raw_data %>% clean_names()


#select important columns: Use Domain knowledge  
data <-  raw_data %>% select( id,los, severity, black, white, asian, latino,
                              mi, pvd, chf, cvd, dement, copd, dm_complicated, dm_simple,
                              renal_disease, stroke, seizure, old_syncope, age_score, o2_sat_94, 
                              temp_38, map_70, d_dimer_yes, plts_score, inr_yes, bun_30, crtn_score,
                              sodium_139_or_154, ast_40, alt_40, wbc_1_8_or_4_8, lymphocytes_1, il6_150,
                              ferritin_300, c_reactive_prot_10, procalciton_0_1, troponin_0_1, death
                              )



# Split the data into train (80%) and test (20%) sets
set.seed(321)
trainIndex <- createDataPartition(data$death, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]


# Logistic regression model
log_model <- glm(death ~ ., data = trainData %>% select(-id), family = binomial)

# Summarize the model
summary(log_model)

# Calculate McFadden's R^2
log_likelihood_full <- logLik(log_model)
log_likelihood_null <- logLik(update(log_model, . ~ 1))

mcfadden_r2 <- 1 - (log_likelihood_full / log_likelihood_null)
print(mcfadden_r2)



# Metrics train data
log_train_pred <- predict(log_model, newdata = trainData, type = "response")
log_train_class <- ifelse(log_train_pred > 0.5, 1, 0)
confusionMatrix(as.factor(log_train_class), as.factor(trainData$death))



# Predictions on test data
log_pred <- predict(log_model, newdata = testData, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

#create new df with new columns -- log_pred and log_pred_class
testDataNewDf <- data.frame(testData,log_pred, log_pred_class )


# Evaluate performance
confusionMatrix(as.factor(log_pred_class), as.factor(testData$death))



log_train_accuracy <- mean(log_train_class == trainData$death)
log_test_accuracy <- mean(log_pred_class == testData$death)

cat("Logistic Regression - Train Accuracy:", log_train_accuracy, "\n")
cat("Logistic Regression - Test Accuracy:", log_test_accuracy, "\n")


#######
# Random Forest model
rf_model <- randomForest(as.factor(death) ~ ., data = trainData %>% select(-id), importance = TRUE)

# Variable importance
importance(rf_model)
varImpPlot(rf_model)

# Predictions on test data
rf_pred <- predict(rf_model, newdata = testData, type = "class")

# Evaluate performance
confusionMatrix(rf_pred, as.factor(testData$death))

# Metrics
rf_train_pred <- predict(rf_model, newdata = trainData, type = "class")
rf_train_accuracy <- mean(rf_train_pred == trainData$death)
rf_test_accuracy <- mean(rf_pred == testData$death)

cat("Random Forest - Train Accuracy:", rf_train_accuracy, "\n")
cat("Random Forest - Test Accuracy:", rf_test_accuracy, "\n")



# Decision tree model
dt_model <- rpart(as.factor(death) ~ ., data = trainData %>% select(-id), method = "class")

# Plot the tree
plot(dt_model)
text(dt_model)

# Predictions on test data
dt_pred <- predict(dt_model, newdata = testData, type = "class")

# Evaluate performance
confusionMatrix(dt_pred, as.factor(testData$death))

# Metrics
dt_train_pred <- predict(dt_model, newdata = trainData, type = "class")
dt_train_accuracy <- mean(dt_train_pred == trainData$death)
dt_test_accuracy <- mean(dt_pred == testData$death)

cat("Decision Tree - Train Accuracy:", dt_train_accuracy, "\n")
cat("Decision Tree - Test Accuracy:", dt_test_accuracy, "\n")



# XGBoost
train_matrix <- model.matrix(death ~ . - 1, data = trainData)
train_label <- as.numeric(as.character(trainData$death))
test_matrix <- model.matrix(death ~ . - 1, data = testData)
test_label <- as.numeric(as.character(testData$death))


xgb_model <- xgboost(data = train_matrix, label = train_label, nrounds = 100, objective = "binary:logistic", verbose = 0,
                     eta = 0.1,   # Learning rate
                     max_depth = 3)



xgb_pred <- predict(xgb_model, test_matrix)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)


# Make predictions on test data
xgb_pred <- predict(xgb_model, newdata = test_matrix)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)  # Convert probabilities to binary class labels

# Evaluate performance with confusion matrix
confusionMatrix(as.factor(xgb_pred_class), as.factor(test_label))

# Calculate accuracy metrics
xgb_train_pred <- predict(xgb_model, newdata = train_matrix)
xgb_train_pred_class <- ifelse(xgb_train_pred > 0.5, 1, 0)
xgb_train_accuracy <- mean(xgb_train_pred_class == train_label)
xgb_test_accuracy <- mean(xgb_pred_class == test_label)

# Display results
cat("XGBoost Model - Train Accuracy:", xgb_train_accuracy, "\n")
cat("XGBoost Model - Test Accuracy:", xgb_test_accuracy, "\n")



# Create a new dataset (using some values from the original test set)
new_data <- testData[1:200, ]
new_data$death <- NULL

# Logistic regression prediction
log_new_pred <- predict(log_model, newdata = new_data, type = "response")


# Random Forest prediction
rf_new_pred <- predict(rf_model, newdata = new_data, type = "prob")

# Decision Tree prediction
dt_new_pred <- predict(dt_model, newdata = new_data, type = "prob")



#generate new data frame
log_new_pred <- data.frame(log_new_pred)
log_new_pred$id <- as.numeric(rownames(log_new_pred))
log_new_pred$logclass <- ifelse(log_new_pred$log_new_pred > 0.5, 1, 0)


rf_new_pred <- as.data.frame(rf_new_pred)
rf_new_pred$id <- as.numeric(rownames(rf_new_pred))
rf_new_pred <- rf_new_pred %>% select(-`0`) %>% rename(rf_pred =`1`)


dt_new_pred <- as.data.frame(dt_new_pred)
dt_new_pred$id <- as.numeric(rownames(dt_new_pred))
dt_new_pred <- dt_new_pred %>% select(-`0`) %>% rename(dt_pred =`1`)



new_data_all <- new_data %>%
  left_join(rf_new_pred, by = "id") %>%
  left_join(dt_new_pred, by = "id") %>%
  left_join(log_new_pred, by = "id")




#######
#Test prediction with new_test_data similar to raw data
set.seed(123)
new_test_data <- data %>%
  sample_n(450)

#select vars used for prediction
#new_test_data_preproc <- new_test_data[, selected_vars]


# Apply the transformations to new_testData
#new_test_data_preproc[, cont_vars] <- predict(preProcValues, new_test_data_preproc[, cont_vars])


# One-hot encode categorical variables
#new_test_data_preproc <- data.frame(predict(DummyModel, newdata = new_test_data_preproc))


# Predictions on test data
log_pred <- predict(log_model, newdata = new_test_data, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

#create new df with new columns -- log_pred and log_pred_class
new_test_data_pred_df <- data.frame(new_test_data,log_pred, log_pred_class )

# Evaluate performance   #This step wont likely be needed since no death in expected data
#confusionMatrix(as.factor(log_pred_class), as.factor(new_test_data_preproc$death))  
confusionMatrix(as.factor(new_test_data_pred_df$death), as.factor(new_test_data_pred_df$log_pred_class))





