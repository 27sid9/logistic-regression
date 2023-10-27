# logistic-regression
---
title: "R Notebook"
output: html_notebook
---



```{r}
rm(list = ls())
```

```{r}
library(tidyverse) 
library(caret) 
library(mlbench) 
theme_set(theme_bw())
```

```{r}
# Load the data and remove NAs
df <- read.csv("CLABSI NEW1 data.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

df <- na.omit(df)
# Inspect the data
sample_n(df, 3)
# Split the data into training and test set
set.seed(123)
training.samples <- df$EVENT %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df[training.samples, ]
test.data <- df[-training.samples, ]
```
```{r}
# Fit the model
model <- glm(EVENT ~ Gender + AGE + Post.procedure_BSI + Procedure.Code + 
             Location + hemodialysis_catheter + Extracorporeal_life + 
             Ventricular.assist_device + Munchausen.Syndrome_admission +
             Observed.suspected_vascular.line_BSI + Signs...Symptoms +
             Died..Yes.No. + BSI_Death + COVID.19..Yes.No. ,
             data = df, family = binomial)
# Summarize the model
summary(model)
# Make predictions
probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy
mean(predicted.classes == test.data$EVENT)
```
```{r}
# Assuming your original data is in the 'data' dataframe
set.seed(42) # For reproducibility
train_indices <- sample(nrow(df), 0.8 * nrow(df)) # 80% for training
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]


# Train the model on the training set
model <- glm(EVENT ~ Gender + AGE + Post.procedure_BSI + Procedure.Code + 
             Location + hemodialysis_catheter + Extracorporeal_life + 
             Ventricular.assist_device + Munchausen.Syndrome_admission +
             Observed.suspected_vascular.line_BSI + Signs...Symptoms +
             Died..Yes.No. + BSI_Death + COVID.19..Yes.No.,
             data = train_data, family = binomial)

# Get predicted probabilities and classes for the test set
test_data$predicted_probs <- predict(model, test_data, type = "response")
test_data$predicted_classes <- ifelse(test_data$predicted_probs >= 0.5, 1, 0)

# Calculate accuracy
accuracy <- sum(test_data$EVENT == test_data$predicted_classes) / nrow(test_data)
print(accuracy)


```


```{r}
model <- glm( EVENT ~ Gender + AGE + Post.procedure_BSI + Procedure.Code + 
             Location + hemodialysis_catheter + Extracorporeal_life + 
             Ventricular.assist_device + Munchausen.Syndrome_admission +
             Observed.suspected_vascular.line_BSI + Signs...Symptoms +
             Died..Yes.No. + BSI_Death + COVID.19..Yes.No. ,, data = train.data, family = binomial)
summary(model)$coef
```
```{r}
# Step 3: Get predicted probabilities
df$predicted_probs <- predict(model, df, type = "response")

# Assuming the actual class labels are in the 'EVENT' column of the 'data' data frame
df$predicted_classes <- ifelse(df$predicted_probs >= 0.3, 1, 0)

# Calculate accuracy
correct_predictions <- sum(df$predicted_classes == df$EVENT)
total_predictions <- nrow(df)
accuracy <- correct_predictions / total_predictions

```
```{r}
# Create a new variable to hold the logistic regression results
results_data <- data.frame(df, predicted_classes = df$predicted_classes)

# Convert "EVENT" and "predicted_classes" to factors with the same levels
results_data$EVENT <- factor(results_data$EVENT, levels = c(0, 1))
results_data$predicted_classes <- factor(results_data$predicted_classes, levels = c(0, 1))

# Create the confusion matrix
conf_matrix <- confusionMatrix(results_data$EVENT, results_data$predicted_classes)

# Print the confusion matrix
print(conf_matrix)



```
```{r}
# Access the false positive (FP) and false negative (FN) counts
FP <- conf_matrix$table[1, 2]  # False positive count
FN <- conf_matrix$table[2, 1]  # False negative count

# Print the false positive and false negative counts
cat("False Positive (FP):", FP, "\n")
cat("False Negative (FN):", FN, "\n")

```
```{r}
# Example new data with predictor variables (replace with your actual data)
newdata <- data.frame(Gender = c(0, 1),
                      AGE = c(50, 60),
                      Post.procedure_BSI = c(0, 1),
                      Procedure.Code = c(1, 3),
                      Location = c(4, 6),
                      hemodialysis_catheter = c(0, 1),
                      Extracorporeal_life = c(0, 0),
                      Ventricular.assist_device = c(0, 0),
                      Munchausen.Syndrome_admission = c(0, 1),
                      Observed.suspected_vascular.line_BSI = c(0, 0),
                      Signs...Symptoms = c(1, 0),
                      Died..Yes.No. = c(0, 0),
                      BSI_Death = c(0, 0),
                      COVID.19..Yes.No. = c(0, 1))

# Perform predictions
newdata$predicted_probs <- predict(model, newdata, type = "response")
newdata$predicted_classes <- ifelse(newdata$predicted_probs >= 0.5, 1, 0)

# View the predictions
head(newdata)

```


```{r}
# Create newdata with the required predictor variables
newdata <- data.frame(
  Gender = c(0, 1),                      # Replace with the correct values
  AGE = c(0, 1),                       # Replace with the correct values
  Post.procedure_BSI = c(0, 1),          # Replace with the correct values
  Procedure.Code = c(4, 6),              # Replace with the correct values
  Location = c(1, 3),                    # Replace with the correct values
  hemodialysis_catheter = c(1, 0),       # Replace with the correct values
  Extracorporeal_life = c(0, 1),         # Replace with the correct values
  Ventricular.assist_device = c(1, 0),   # Replace with the correct values
  Munchausen.Syndrome_admission = c(0, 1), # Replace with the correct values
  Observed.suspected_vascular.line_BSI = c(0, 1), # Replace with the correct values
  Signs...Symptoms = c(1, 0),           # Replace with the correct values
  Died..Yes.No. = c(1, 0),              # Replace with the correct values
  BSI_Death = c(1, 0),                  # Replace with the correct values
  COVID.19..Yes.No. = c(0, 1),          # Replace with the correct values
  Pathogens.Identified = c(1, 0)        # Replace with the correct values
)

# Make predictions using the model and newdata
probabilities <- predict(model, newdata, type = "response")


probabilities <- model %>% predict(newdata, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
print(predicted.classes)

```
```{r}
train.data %>%
  mutate(prob = ifelse(EVENT == "pos", 1, 0)) %>%
  ggplot(aes(Gender, prob)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Logistic Regression Model", 
    x = "Plasma Glucose Concentration",
    y = "Probability of being diabete-pos"
  )
```
```{r}
library(ggplot2)
library(dplyr)


N <- 60 # number of points per class
D <- 2 # dimensionality, we use 2D data for easy visualization
K <- 2 # number of classes, binary for logistic regression
X <- data.frame() # data matrix (each row = single example, can view as xy coordinates)
y <- data.frame() # class labels

set.seed(56)

t <- seq(0,1,length.out = N) 
for (j in (1:K)){
  # t, m are parameters of parametric equations x1, x2
  # t <- seq(0,1,length.out = N) 
  # add randomness 
  m <- rnorm(N, j+0.5, 0.25) 
  Xtemp <- data.frame(x1 = 3*t , x2 = m - t) 
  ytemp <- data.frame(matrix(j-1, N, 1))
  X <- rbind(X, Xtemp)
  y <- rbind(y, ytemp)
}

data <- cbind(X,y)
colnames(data) <- c(colnames(X), 'label')
```

```{r}
# lets visualize the data:
data_plot <- ggplot(data) + geom_point(aes(x=x1, y=x2, color = as.character(label)), size = 2) + 
  scale_colour_discrete(name  ="Label") + 
  ylim(0, 3) + coord_fixed(ratio = 1) +
  ggtitle('Data to be classified') +
  theme_bw(base_size = 12) +
  theme(legend.position=c(0.85, 0.87))
```

```{r}
#png(file.path('images', 'data_plot.png'))
print(data_plot)
#dev.off()
```
```{r}
#sigmoid function, inverse of logit
sigmoid <- function(z){1/(1+exp(-z))}

```


```{r}
cost <- function(theta, X, y){
  m <- length(y) # number of training examples
  h <- sigmoid(X %*% theta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m
  J
}
```

```{r}
#gradient function
grad <- function(theta, X, y){
  m <- length(y) 
  
  h <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(h - y))/m
  grad
}
```

```{r}
logisticReg <- function(X, y){
  #remove NA rows
  X <- na.omit(X)
  y <- na.omit(y)
  #add bias term and convert to matrix
  X <- mutate(X, bias =1)
  #move the bias column to col1
  X <- as.matrix(X[, c(ncol(X), 1:(ncol(X)-1))])
  y <- as.matrix(y)
  #initialize theta
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X))
  #use the optim function to perform gradient descent
  costOpti <- optim(theta, fn = cost, gr = grad, X=X, y=y)
  #return coefficients
  return(costOpti$par)
}
```

```{r}
logisticProb <- function(theta, X){
  X <- na.omit(X)
  #add bias term and convert to matrix
  X <- mutate(X, bias =1)
  X <- as.matrix(X[,c(ncol(X), 1:(ncol(X)-1))])
  return(sigmoid(X%*%theta))
}
```

```{r}

logisticPred <- function(prob){
  return(round(prob, 0))
}
```

```{r}
# training
theta <- logisticReg(X, y)
prob <- logisticProb(theta, X)
pred <- logisticPred(prob)

```

```{r}
# generate a grid for decision boundary, this is the test set
grid <- expand.grid(seq(0, 3, length.out = 100), seq(0, 3, length.out = 100))
# predict the probability
probZ <- logisticProb(theta, grid)
# predict the label
Z <- logisticPred(probZ)
gridPred = cbind(grid, Z)
```

```{r}
# decision boundary visualization
p <- ggplot() + geom_point(data = data, aes(x=x1, y=x2, color = as.character(label)), size = 2, show.legend = F) + 
  geom_tile(data = gridPred, aes(x = grid[, 1],y = grid[, 2], fill=as.character(Z)), alpha = 0.3, show.legend = F)+ 
  ylim(0, 3) +
  ggtitle('Decision Boundary for Logistic Regression') +
  coord_fixed(ratio = 1) +
  theme_bw(base_size = 12) 

#png(file.path('images', 'logistic_regression.png'))
print(p)
```

