# Load required packages
library(tidymodels)
library(palmerpenguins)  # Contains the penguins dataset

# Set seed for reproducibility
set.seed(1234)

# Split the data into training (70%) and testing (30%) sets
penguins_split <- initial_split(penguins, prop = 0.7, strata = species)

# Extract training and testing datasets
penguins_train <- training(penguins_split)
penguins_test <- testing(penguins_split)

# Create a 10-fold cross-validation dataset from training data
penguins_folds <- vfold_cv(penguins_train, v = 10)

# Print structure to confirm setup
glimpse(penguins_train)
glimpse(penguins_test)
print(penguins_folds)


