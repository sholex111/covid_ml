
## Test the model on seen and unseen data
#Run after models have been built


# Create a new dataset (using some values from the original test set)

#new_data <- testData[1:450, ]
set.seed(1000)
new_data <- testData %>% sample_n(200)
#new_data$death <- NULL

# Logistic regression prediction
log_new_pred <- predict(log_model, newdata = new_data, type = "response")


# Random Forest prediction
rf_new_pred <- predict(rf_model, newdata = new_data, type = "prob")

# Decision Tree prediction
dt_new_pred <- predict(dt_model, newdata = new_data, type = "prob")



#generate new data frame
log_new_pred <- data.frame(log_new_pred)
log_new_pred$id <- new_data$id
log_new_pred$logclass <- ifelse(log_new_pred$log_new_pred > 0.5, 1, 0)


rf_new_pred <- as.data.frame(rf_new_pred)
rf_new_pred$id <- new_data$id
rf_new_pred <- rf_new_pred %>% select(-`0`) %>% rename(rf_pred =`1`)


dt_new_pred <- as.data.frame(dt_new_pred)
dt_new_pred$id <- new_data$id
dt_new_pred <- dt_new_pred %>% select(-`0`) %>% rename(dt_pred =`1`)



new_data_all <- new_data %>%
  left_join(rf_new_pred, by = "id") %>%
  left_join(dt_new_pred, by = "id") %>%
  left_join(log_new_pred, by = "id")


confusionMatrix(as.factor(new_data_all$death), as.factor(new_data_all$logclass), positive = "1")





#######
#Test prediction with new_test_data similar to raw data
set.seed(4321)
new_test_data <- raw_data_stripped %>%
  sample_n(2000)

#select vars used for prediction
#new_test_data_preproc <- new_test_data[, selected_vars]
new_test_data_preproc <- new_test_data



# Apply the transformations to new_testData
new_test_data_preproc[, cont_vars] <- predict(preProcValues, new_test_data_preproc[, cont_vars])


# One-hot encode categorical variables
#new_test_data_preproc <- data.frame(predict(DummyModel, newdata = new_test_data_preproc))


# Predictions on test data
log_pred <- predict(log_model, newdata = new_test_data, type = "response")
log_pred_class <- ifelse(log_pred > 0.50, 1, 0)

#create new df with new columns -- log_pred and log_pred_class
new_test_data_pred_df <- data.frame(new_test_data,log_pred, log_pred_class )

# Evaluate performance   #This step wont likely be needed since no death in expected data
#confusionMatrix(as.factor(log_pred_class), as.factor(new_test_data_preproc$death))  
confusionMatrix(as.factor(new_test_data_pred_df$death), as.factor(new_test_data_pred_df$log_pred_class), positive = "1")





