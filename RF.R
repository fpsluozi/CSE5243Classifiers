# Author: Yiran Luo
# Last modified: Nov 3, 2015
# Title: Training a Random Forest classifier with Wine data

library(e1071)

library(corrplot)  # graphical display of the correlation matrix
library(rpart)     # decision tree 
library(caret)     # classification and regression training
library(klaR)      # naive bayes
library(nnet)      # neural networks (nnet and avNNet)
library(kernlab)   # support vector machines (svmLinear and svmRadial)
library(randomForest)  # random forest, also for recursive feature elimination
library(gridExtra) # save dataframes as images
library(pROC)

path <- “~/“
model_name <- "rf-"

# Load the wine dataset
wine <- read.csv(“~/wine.csv", header = TRUE, sep = ',')

df <- data.frame(wine)
set.seed(1234) 

# Data split into 30/70
trainIndices <- createDataPartition(df$class, p = 0.7, list = FALSE)
train <- df[trainIndices, ]
test <- df[-trainIndices, ]

# Using 10-fold CV for training the model
set.seed(2333)
fit_control <- trainControl(method = 'cv', number = 10, 
                            summaryFunction = twoClassSummary, 
                            classProbs = T,
                            savePredictions = T)

train <- train[,!(names(train) %in% c('quality', 'ID'))]
test <- test[,!(names(test) %in% c('quality', 'ID'))]

# Train the model and do predictions
fit_model <- train(class ~ ., data = train,
                method ='rf',
                trControl = fit_control,
                preProces = c("scale", "center"),
                metric = "ROC")
predict_model <- predict(fit_model, newdata = test, type="prob")
predict_model_raw <- predict(fit_model, newdata = test)
confmat_model <- confusionMatrix(predict_model_raw, test$class, positive = 'High')
importance_model <- varImp(fit_model, scale = TRUE)

# Feature importance
png(paste0(path, model_name, 'importance.png'))
plot(importance_model, main = 'Feature importance for Random Forest')
dev.off()

# Output the confusion matrix and reports
sink(paste0(path, model_name, "report.txt"))
print(fit_model)
print(confmat_model)
sink()

# Output the ROC curve
png(paste0(path, model_name, 'roc.png'))
roc_curve <- roc(response = test$class, predictor = predict_model[,1], levels = rev(levels(test$class)))
plot(roc_curve, xlab = "Specificity (1 - fpr)", ylab = "Sensitivity (tpr)", main = "ROC Curve for Random Forest (randomForest)")
dev.off()