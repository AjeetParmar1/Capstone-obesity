library(tidyverse)
library(caret)
library(gam)
library(rpart)
library(rattle) # library to generate clean looking trees

# suppresses warnings about sample.kind = "Rounding"
suppressWarnings(set.seed(1, sample.kind = "Rounding"))

# reads in the data. 
data <- read_csv("diabetes_data_upload.csv")
data <- data %>% as_data_frame()

# I manually convert the qualitative data to factors. 
data$Polyuria <- data$Polyuria %>% factor(levels = c("No", "Yes"))
data$Polydipsia <- data$Polydipsia %>% factor(levels = c("No", "Yes"))
data$`sudden weight loss` <- data$`sudden weight loss` %>% factor(levels = c("No", "Yes"))
data$weakness <- data$weakness%>% factor(levels = c("No", "Yes"))
data$Polyphagia<- data$Polyphagia %>% factor(levels = c("No", "Yes"))
data$`Genital thrush` <- data$`Genital thrush` %>% factor(levels = c("No", "Yes"))
data$`visual blurring` <- data$`visual blurring`%>% factor(levels = c("No", "Yes"))
data$Polydipsia <- data$Polydipsia %>% factor(levels = c("No", "Yes"))
data$Itching<- data$Itching %>% factor(levels = c("No", "Yes"))
data$Itching <- data$Itching%>% factor(levels = c("No", "Yes"))
data$Irritability <- data$Irritability %>% factor(levels = c("No", "Yes"))
data$`delayed healing` <- data$`delayed healing` %>% factor(levels = c("No", "Yes"))
data$`partial paresis`<- data$`partial paresis` %>% factor(levels = c("No", "Yes"))
data$`muscle stiffness` <- data$`muscle stiffness` %>% factor(levels = c("No", "Yes"))
data$Alopecia <- data$Alopecia %>% factor(levels = c("No", "Yes"))
data$Obesity <- data$Obesity %>% factor(levels = c("No", "Yes"))
data$class <- data$class %>% factor(levels = c("Negative", "Positive"))

# Splits the data into training and testing data.
index <- createDataPartition(y = data$Age, times = 1, p = 0.6, list = FALSE)
training_data <- data %>% slice(index)
testing_data <- data %>% slice(-index)

# determining cases by gender
data %>% ggplot(aes(Gender, fill = class, color = class)) +
  geom_bar()

# creating a graph of cases by polyuria
data %>% ggplot(aes(Polyuria, fill = class, color = class)) +
  geom_bar()

# determining cases by polydipsia state. 
data %>% ggplot(aes(Polydipsia, fill = class, color = class)) +
  geom_bar()

# determining cases by if the patient suffers from sudden weight loss. 
data %>% ggplot(aes(`sudden weight loss`, fill = class, color = class)) +
  geom_bar()

# cases by state of weakness.
data %>% ggplot(aes(weakness, fill = class, color = class)) +
  geom_bar()

# cases by polyphagia
data %>% ggplot(aes(Polyphagia, fill = class, color = class)) +
  geom_bar()

# cases by genital thrush
data %>% ggplot(aes(`Genital thrush`, fill = class, color = class)) +
  geom_bar()

# cases by visual blurring
data %>% ggplot(aes(`visual blurring`, fill = class, color = class)) +
  geom_bar()

# cases by itching
data %>% ggplot(aes(Itching, fill = class, color = class)) +
  geom_bar()

# cases by irritability
data %>% ggplot(aes(Irritability, fill = class, color = class)) +
  geom_bar()

# cases by delayed healing
data %>% ggplot(aes(`delayed healing`, fill = class, color = class)) +
  geom_bar()

# cases by partial paresis
data %>% ggplot(aes(`partial paresis`, fill = class, color = class)) +
  geom_bar()

# cases by muscle stiffness
data %>% ggplot(aes(`muscle stiffness`, fill = class, color = class)) +
  geom_bar()

# cases by alopecia
data %>% ggplot(aes(Alopecia, fill = class, color = class)) +
  geom_bar()

# cases by obesity
data %>% ggplot(aes(Obesity, fill = class, color = class)) +
  geom_bar()

# Generates the histogram of people with diabetes by age. 
options(warn=-1)
data %>% group_by(Age) %>% mutate(numwith = sum(class == "Positive")) %>% 
  ggplot(aes(x = Age)) + geom_histogram(color = "blue", bins = 10) + 
  ylab("Number of people with diabetes") + xlab("Age") + 
  ggtitle("Number of people with diabetes by age")

# Logistic regression model 
glm_train <- train(class ~., data = training_data, method = "glm", family = "binomial")
y_hat_glm <- predict(glm_train, testing_data)
#predicting the accuracy given our logistic regression model. 
glm_acc <- mean(y_hat_glm == testing_data$class)

# LDA regression model 
lda_train <- train(class ~., data = training_data, method = "lda", family = "binomial")
y_hat_lda <- predict(lda_train, testing_data)
#predicting the accuracy given our LDA model. 
lda_acc <- mean(y_hat_lda == testing_data$class)

# QDA regression model 
qda_train <- train(class ~., data = training_data, method = "qda", family = "binomial")
y_hat_qda <- predict(qda_train, testing_data)
#predicting the accuracy given our QDA model. 
qda_acc <- mean(y_hat_qda == testing_data$class)

# Naive Bayes regression model 
Bayes_train <- train(class ~., data = training_data, method = "naive_bayes")
y_hat_Bayes <- predict(Bayes_train, testing_data)
#predicting the accuracy given our Naive Bayes model. 
Bayes_acc <- mean(y_hat_Bayes == testing_data$class)

#Tuning k in knn
ks = seq(1,15, 1)
knn_tune <- train(class ~ ., method = "knn", data = training_data,tuneGrid = data.frame(k = ks))
#plotting my knn results
plot(knn_tune$results$Accuracy)

#Tuning mtry in random forest
mtry_val <- seq(1, 25, 1)
rf_test <- train(class ~ ., method = "rf", data = training_data, tuneGrid = data.frame(mtry = mtry_val))

# print the table
rf_test$results$Accuracy

#Note: for some reason, whenever I plotted it, it would give me a different plot each time.
# So I just printed the table instead. 

# Using the results created, I create my KNN model and the random forest model.

knn_fit <- train(class ~ ., method = "knn", data = training_data, tuneGrid = data.frame(k = 1))
y_hat_knn <- predict(knn_fit, testing_data)
# determining the accuracy of my knn model. 
knn_acc <- mean(y_hat_knn == testing_data$class)

rf <- train(class ~ ., method = "rf", data = training_data, tuneGrid = data.frame(mtry = 4))
y_hat_rf <- predict(rf, testing_data)
# finding the overall accuracy of my random forest model. 
rf_acc <- mean(y_hat_rf == testing_data$class)

tree <- rpart(class~., data = training_data) # generate the tree

fancyRpartPlot(tree, caption = "decision tree for the data") # creates the tree for the training data.

#Creating the ensemble as a data frame
ensemble <- data.frame(LDA = y_hat_lda, QDA = y_hat_qda,
                       Naive_Bayes = y_hat_Bayes, knn = y_hat_knn, 
                       rf = y_hat_rf, glm = y_hat_glm)

# Ultimatum helps us determine whether a test is positive. 
# If most claim that it is a positive test, we classify it as a positive case. 
# otherwise, we classify it as a negative result. 
ultimatum <- rowMeans(ensemble == "Positive")
y_hat <- ifelse(ultimatum > 0.5, "Positive", "Negative")
#ensemble_acc gives us our overall accuracy
ensemble_acc <- mean(y_hat == testing_data$class)

# all accuracies of each algorithm and the ensemble are compiled into a data frame. 
data.frame(GLM = glm_acc, LDA = lda_acc, QDA = qda_acc, Naive_Bayes = Bayes_acc, knn = knn_acc, 
           Random_Forest = rf_acc, Ensemble = ensemble_acc)