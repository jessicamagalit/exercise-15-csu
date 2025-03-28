## Exercise 15
# Load required libraries
library(tidymodels)
library(nnet)  # For multinom_reg()
install.packages("ranger")  # Install ranger if missing
library(ranger)

# Load dataset
data(penguins, package = "palmerpenguins")

# Remove NA values
penguins <- na.omit(penguins)

# Set a seed for reproducibility
set.seed(123)

# Split the data into training (70%) and testing (30%)
penguin_split <- initial_split(penguins, prop = 0.7, strata = species)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

# Create 10-fold cross-validation from training data
penguin_folds <- vfold_cv(penguin_train, v = 10, strata = species)

## Model Fitting and Workflow

# Define multinomial logistic regression model (for multiclass)
log_reg_model <- multinom_reg() %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Define random forest model
rand_forest_model <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Define a recipe (preprocessing steps)
penguin_recipe <- recipe(species ~ ., data = penguin_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

# Create workflows for both models
log_reg_workflow <- workflow() %>%
  add_model(log_reg_model) %>%
  add_recipe(penguin_recipe)

rand_forest_workflow <- workflow() %>%
  add_model(rand_forest_model) %>%
  add_recipe(penguin_recipe)

# Fit models using resampling
log_reg_results <- fit_resamples(log_reg_workflow, resamples = penguin_folds, metrics = metric_set(accuracy))
rand_forest_results <- fit_resamples(rand_forest_workflow, resamples = penguin_folds, metrics = metric_set(accuracy))

# Collect results
log_reg_metrics <- collect_metrics(log_reg_results)
rand_forest_metrics <- collect_metrics(rand_forest_results)

# Compare models
log_reg_metrics
rand_forest_metrics

# Comment: 
# Based on accuracy, I believe the best model for predicting species in this dataset is the logistics regression model.
