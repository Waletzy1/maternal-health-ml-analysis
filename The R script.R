# R code 
library(ggplot2)
library(factoextra)
library(caret)
library(dplyr)
library(nnet)
library(caret)
library(randomForest)
library(tidyverse)
library(corrplot)
library(DataExplorer)

Risks_dataset <- read.csv("Health_risk.csv")
Factors_dataset <- read.csv("Maternal_health_factor.csv")

summary(Risks_dataset)
summary(Factors_dataset)


##To clear the missing and undefined values in the datastes
colSums(is.na(Risks_dataset))
colSums(is.na(Factors_dataset))

Factors_clean <- na.omit(Factors_dataset)  


##To visualize the distribution of both the datasets:
Risks_dataset %>%
  gather(key = "Feature", value = "Value", -RiskLevel) %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.5) +
  facet_wrap(~ Feature, scales = "free") +
  theme_minimal() +
  labs(title = "Histogram of Variables distributions in Risks dataset")

Factors_dataset %>%
  gather(key = "Feature", value = "Value", -RiskLevel) %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "green", alpha = 0.5) +
  facet_wrap(~ Feature, scales = "free") +
  theme_minimal() +
  labs(title = "Histogram of Variables Distributions in Factors Dataset")


##The correlation marix of both the datasets features
cor_matrix1 <- cor(Risks_dataset %>% select_if(is.numeric))
cor_matrix2 <- cor(Factors_clean %>% select_if(is.numeric))

plot.new()
dev.off()
corrplot(cor_matrix1, method = "color", addCoef.col = "black", 
         tl.col = "black", tl.srt = 45, 
         title = "Correlation Matrix - Risks dataset")

plot.new()
dev.off()
windows(width = 10, height = 10)
corrplot(cor_matrix2, method = "color", addCoef.col = "black", 
         tl.col = "black", tl.srt = 45, 
         title = "Correlation Matrix - Factors dataset")


##Task 1: Unsupervised learning: Clustering
Clustering_df <- read.csv("Health_risk.csv")
str(Clustering_df)
fviz_nbclust(Clustering_df, kmeans, method = "wss")
Clustering_data_scaled <- scale(data[, -1]) 

set.seed(123)
kmeans_result <- kmeans(Clustering_data_scaled, centers = 3)

data$cluster <- kmeans_result$cluster
cluster_colors <- c("red", "blue", "green")

fviz_cluster(kmeans_result, data = Clustering_data_scaled, 
             ellipse.type = "convex", palette = cluster_colors)

Clustering_distance_matrix <- dist(Clustering_data_scaled)
hclust_result <- hclust(Clustering_distance_matrix, method = "ward.D2")

plot(hclust_result)

data$cluster_hc <- cutree(hclust_result, k = 3)
fviz_cluster(list(data = Clustering_data_scaled, cluster = data$cluster_hc), geom = "point", palette = cluster_colors)


##Task 2: Regression
Regression_df <- read.csv("Maternal_health_factor.csv")

str(Regression_df)

Regression_df_clean <- na.omit(Regression_df)

str(Regression_df_clean)

model_linear <- lm(BloodSugar ~ Age + SystolicBP + DiastolicBP + 
                     BodyTemp + BMI + Previous_Complications + 
                     Preexisting_Diabetes + Gestational_Diabetes + 
                     HeartRate + RiskLevel, data = Regression_df_clean)

model_poly <- lm(BloodSugar ~ poly(Age, 2) + poly(SystolicBP, 2) + 
                   poly(DiastolicBP, 2) + poly(BodyTemp, 2) + 
                   poly(BMI, 2) + Previous_Complications + 
                   Preexisting_Diabetes + Gestational_Diabetes + 
                   HeartRate + RiskLevel, data = Regression_df_clean)

anova(model_linear, model_poly)

set.seed(123)
trainIndex <- createDataPartition(Regression_df_clean$BloodSugar, p = 0.8, list = FALSE)
trainData <- Regression_df_clean[trainIndex, ]
testData <- Regression_df_clean[-trainIndex, ]

model_poly <- lm(BloodSugar ~ poly(Age, 2) + poly(SystolicBP, 2) + 
                   poly(DiastolicBP, 2) + poly(BodyTemp, 2) + 
                   poly(BMI, 2) + Previous_Complications + 
                   Preexisting_Diabetes + Gestational_Diabetes + 
                   HeartRate + RiskLevel, data = trainData)

summary(model_poly)


predictions <- predict(model_poly, newdata = testData)

mse <- mean((testData$BloodSugar - predictions)^2)
r_squared <- summary(model_poly)$r.squared

cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r_squared, "\n")

ggplot(data.frame(Actual = testData$BloodSugar, Predicted = predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "red") +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid") +
  labs(title = "Actual vs Predicted Maternal Blood Sugar Levels", x = "Actual", y = "Predicted") +
  theme_minimal()


##Task 3: Classification
Classification_df <- read.csv("Health_risk.csv")

Classification_df$RiskLevel <- as.factor(data$RiskLevel)
set.seed(123)
trainIndex <- createDataPartition(Classification_df$RiskLevel, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

set.seed(123)
rf_model <- randomForest(RiskLevel ~ ., data = trainData, ntree = 100, mtry = 3, importance = TRUE)

predictions <- predict(rf_model, testData)

conf_matrix <- confusionMatrix(predictions, testData$RiskLevel)
print(conf_matrix)

varImpPlot(rf_model)


## The open access links to the datasets

* [Risks_dataset](https://archive.ics.uci.edu/dataset/863/maternal%2Bhealth%2Brisk?utm_source=chatgpt.com)
* [Factors_dataset](https://data.mendeley.com/datasets/p5w98dvbbk/1/files/e7b178fd-09ae-4859-b4b3-5675f79f470c)
