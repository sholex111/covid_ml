#Exploration script

# Load necessary libraries
library(dplyr)
library(tidyr)
library(broom)
library(purrr)
library(corrplot)


#duplicate original variable
raw_data_fine <- raw_data


# Convert categorical variables to factors
categorical_vars <- c("age_5","los_y", "black", "white", "asian", "latino", 
                      "mi", "pvd", "chf", "cvd", "dement", 
                      "copd", "dm_complicated", "dm_simple", 
                      "renal_disease", "all_cns", "pure_cns", 
                      "stroke", "seizure", "old_syncope", 
                      "old_other_neuro", "temp_38", "o2_sat_94", 
                      "map_70", "bun_30", "crtn_score", 
                      "sodium_139_or_154", "glucose_60_or_500", 
                      "ast_40", "alt_40", "wbc_1_8_or_4_8", 
                      "lymphocytes_1", "il6_150", "ferritin_300", 
                      "c_reactive_prot_10","crtn_yes", "procalciton_0_1", 
                      "troponin_0_1", "d_dimer_yes", "d_dimer_3","glucose_yese",
                      "inr_yes", "inr_1_2", "sodimu_yes", "lympho_yes",
                      "il6yes", "pro_cal_c_yes", "trop_yes", "ast_yes",
                      "alt_yes", "wbc_yes", "o2sats_yes","age_score", "plts_score")


# Apply the conversion
raw_data_fine[categorical_vars] <- lapply(raw_data_fine[categorical_vars], as.factor)


#####
# Identify continuous variables
continuous_vars <- names(raw_data_fine)[sapply(raw_data_fine, is.numeric)]  # Get all numeric columns


# Create Table 1
table_1 <- raw_data_fine %>%
  select(death, all_of(continuous_vars)) %>% 
  select(-id, -map_yes, -crct_prot_yes, -ferritin_yes, -temp_yes, -bun_yes ) %>% 
  gather(key = "variable", value = "value", -death) %>%
  group_by(variable) %>%
  summarise(
    mean_death_0 = mean(value[death == 0], na.rm = TRUE),
    mean_death_1 = mean(value[death == 1], na.rm = TRUE),
    t_test = list(t.test(value ~ death)),
    p_value = map_dbl(t_test, ~ .x$p.value)  # Extract the p-value correctly
  ) %>%
  select(variable, mean_death_0, mean_death_1, p_value)



# Round the values to 4 decimal places in the final table and add 'Significant' column
table_1 <- table_1 %>%
  mutate(
    mean_death_0 = round(mean_death_0, 4),
    mean_death_1 = round(mean_death_1, 4),
    p_value = round(p_value, 4),
    significant = if_else(p_value < 0.05, "***", "N")
  )

# View the results
print(table_1)

######


## Table 2 and 3 preserved because they are not yet quite perfect
##### 
##Table_2
# Step 1: Create the original table with counts, percentages, and chi-squared p-values
table_2 <- raw_data_fine %>%
  select(death, all_of(categorical_vars)) %>%
  gather(key = "variable", value = "category", -death) %>%
  group_by(variable, category, death) %>%
  summarise(count = n(), .groups = "drop") %>%
  spread(key = death, value = count, fill = 0) %>%  # Ensure 0 for missing categories
  group_by(variable) %>%
  mutate(
    `0` = as.numeric(`0`),  # Convert counts to numeric
    `1` = as.numeric(`1`),
    total = `0` + `1`,  # Calculate the total count across death categories
    percentage_0 = `0` / total * 100,
    percentage_1 = `1` / total * 100
  ) %>%
  ungroup() %>%
  group_by(variable) %>%
  mutate(
    chi_sq_test = list(chisq.test(c(`0`, `1`))),  # Run chi-squared test by variable
    chi_sq_p_value = map_dbl(chi_sq_test, ~ .x$p.value)  # Extract p-value from chi-squared test
  ) %>%
  select(variable, category, `0`, `1`, percentage_0, percentage_1, chi_sq_p_value)



# Step 2: Manipulate the original table to create a final version with rounding and significance marker
table_2 <- table_2 %>%
  mutate(
    percentage_0 = round(percentage_0, 2),
    percentage_1 = round(percentage_1, 2),
    chi_sq_p_value = round(chi_sq_p_value, 4),
    Significant = ifelse(chi_sq_p_value < 0.05, "***", "N")
  )


# View the final table
print(table_2)


#####
#Table 3

# Step 1: Create table_2 with counts and percentages
table_3 <- raw_data_fine %>%
  select(death, all_of(categorical_vars)) %>%
  gather(key = "variable", value = "category", -death) %>%
  group_by(variable, category, death) %>%
  summarise(count = n(), .groups = "drop") %>%
  spread(key = death, value = count, fill = 0) %>%
  mutate(
    `0` = as.numeric(`0`),
    `1` = as.numeric(`1`),
    total = `0` + `1`,
    percentage_0 = round((`0` / total) * 100, 2),
    percentage_1 = round((`1` / total) * 100, 2)
  ) %>%
  ungroup()

# Step 2: Run chi-squared tests row by row and add p-values
table_3 <- table_3 %>%
  rowwise() %>%
  mutate(
    chi_sq_test = list(chisq.test(matrix(c(`0`, `1`), ncol = 2))),
    chi_sq_p_value = round(chi_sq_test$p.value, 4),
    Significant = ifelse(chi_sq_p_value < 0.05, "***", "N")
  ) %>%
  ungroup() %>%
  select(variable, category, `0`, `1`, percentage_0, percentage_1, chi_sq_p_value, Significant)

# Display the table
print(table_3, )


#####
## correlation matrix
# check correlation matrix and plot to identify possible autocorrelation

raw_data_cont <- raw_data_fine %>%
  select(death, all_of(continuous_vars)) %>% 
  select(-id, -map_yes, -crct_prot_yes, -ferritin_yes, -temp_yes, -bun_yes )
cor_matrix <- cor(raw_data_cont)


# Set the plot size larger with par(mar) and par(pin) parameters for margins and plot dimensions
par(mar = c(5, 5, 5, 5), pin = c(8, 8))  # Adjust to make the plot area larger

# Create the correlation plot with smaller labels
corrplot(cor_matrix, method = "color", type = "full",
         col = colorRampPalette(c("red", "white", "blue"))(200),
         addCoef.col = "black",  
         tl.col = "black", tl.srt = 45,
         tl.cex = 0.8,  # Adjust to make the labels smaller
         cl.cex = 0.8,  # Adjust to make the color legend smaller
         number.cex = 0.6)  # Size for the correlation coefficient values




