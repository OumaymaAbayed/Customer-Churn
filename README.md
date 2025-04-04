# Customer Churn Prediction - Random Forest Model 📊

This R project predicts customer churn using a Random Forest model. The dataset used contains customer details, and the goal is to predict whether a customer will churn (leave the service) or not.

Dataset Source: Kaggle

## 📦 Installation

To get started, install the required packages by running the following commands in your R console:

```R
install.packages("randomForest")
install.packages("caret")
install.packages(c("rpart", "rpart.plot", "rattle", "RColorBrewer"), quiet = TRUE)
```

Then, load the necessary libraries:

```R
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
```

## 📈 Data Preprocessing

1. **Load the dataset:**
   The dataset `Customer-Churn.csv` is read into R.

```R
data <- read.csv('Customer-Churn.csv')
head(data)
str(data)
```

2. **Remove missing values:**
   Missing values are removed from the dataset.

```R
data <- na.omit(data)
```

3. **Convert categorical variables to factors:**
   The target variable `Churn` is converted to a factor.

```R
data$Churn <- as.factor(data$Churn)
```

4. **Drop the `customerID` column:**
   The `customerID` column is removed as it is not needed for modeling.

```R
data <- data[ , !names(data) %in% c("customerID")]
```

## 🔧 Model Training

1. **Split the data into training and testing sets:**

```R
set.seed(123)
trainIndex <- createDataPartition(data$Churn, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
```

2. **Train the Random Forest model:**

```R
rf_model <- randomForest(Churn ~ ., data = trainData, ntree = 100, mtry = 3)
print(rf_model)
```

3. **Make predictions:**

```R
predictions <- predict(rf_model, newdata = testData)
```

4. **Model Evaluation:**
   The Out-Of-Bag (OOB) estimate of error is 20.6%.

## 📊 Decision Tree Model

We also train a decision tree model to compare results:

1. **Train the decision tree model:**

```R
d_tree <- rpart(Churn ~ ., data = trainData, method = "class")
print(d_tree)
```

2. **Plot the decision tree:**

```R
rpart.plot(d_tree, type = 3, extra = 101, fallen.leaves = TRUE, cex = 0.6)
fancyRpartPlot(d_tree)
```

### 🧐 Insights:
- **Contract Length Matters**: Longer contracts significantly reduce churn.
- **Internet Service**: Customers with DSL or no internet service have lower churn rates compared to those with fiber.
- **Tenure Impact**: Customers with longer tenure are less likely to churn.
- **Tech Support**: Lack of tech support with short tenure increases churn probability.

## 🏅 Model Evaluation

1. **Confusion Matrix for Random Forest:**

```R
conf_matrix <- confusionMatrix(predictions, testData$Churn)
print(conf_matrix)
```

**Results:**
- Overall Accuracy: **80.83%**
- Sensitivity: **90.25%** (non-churn)
- Specificity: **54.82%** (churn)
- Kappa: **0.4785**

## 🔧 Model Tuning

1. **Variable Importance:**

```R
importance <- varImpPlot(rf_model, main = "Variable Importance")
print(importance)
```

2. **ROC and AUC:**

```R
probabilities <- predict(rf_model, newdata = testData, type = "prob")
roc_curve <- roc(testData$Churn, probabilities[,2], levels = rev(levels(testData$Churn)))
plot(roc_curve, col = "blue", main = "ROC Curve")
auc(roc_curve)
```

3. **Hyperparameter Tuning:**
   Cross-validation and hyperparameter tuning are done using the `caret` package.

```R
tuning_grid <- expand.grid(.mtry = c(2, 4, 6, 8, 10))
control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

tuned_rf <- train(Churn ~ ., data = trainData, method = "rf", 
                  metric = "Accuracy", tuneGrid = tuning_grid, 
                  trControl = control)
```

4. **Best Model:**
   After tuning, the model’s accuracy slightly decreased, but specificity improved. However, the initial model remained the best for overall performance.

## 📉 Conclusion

While the tuned model showed slight improvements in some areas, the original Random Forest model remains the best choice for predicting customer churn based on the available data.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🤝 Contributions

Feel free to fork this repo, create an issue, or submit a pull request with your suggestions or improvements!

```

This README contains all necessary details for your R project, including setup, model details, evaluation, and conclusions, with emojis to enhance readability and engagement.
