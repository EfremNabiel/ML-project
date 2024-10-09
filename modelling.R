####  Fitting a CNN on spectogram data  ####
# loading keras
library(keras)
library(tidyverse)
# imorting data
load("./data/spectogram_data.RData")

# preparing data 
train_x <- spectogram_data$train_x
train_y <- spectogram_data$train_y
test_x <- spectogram_data$test_x
test_y <- spectogram_data$test_y
# checking that dimension of training data is correct
dim(train_x)



# predicting on test set
preds <- predict(model, test_x)
# reducing by 1 since keras encodes (0-9) while R (1-10)
pred_y <- apply(preds, MARGIN = 1, FUN = which.max) - 1

# computing proportion correctly predicted in test set
mean(test_y == pred_y)

## misclassified images in test set
test_y[which(test_y != pred_y)]
pred_y[which(test_y != pred_y)]

# most of missclassification in 4 and 13
table(pred_y[which(test_y != pred_y)])

# 4 and 13 overrepresented in predictions
table(test_y)

#### displaying CNN spectograms  ####
# importing libraries
library(torch)
library(torchaudio)

# looking at arbitrary spectogram
example <- test_x[4,,]

# displaying the image
image(example[,ncol(example):1])
spectogram_data$test_species[4]

#baseline model:
model<- keras_model_sequential(input_shape = c(dim(train_x)[2],
                                                        dim(train_x)[3],
                                                        1)) %>%
  layer_batch_normalization() %>%
  # convolutional layers
  layer_conv_2d(filters = 12, kernel_size = c(3,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 12, kernel_size = c(3,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 12, kernel_size = c(3,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 12, kernel_size = c(3,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  # feed-forward network
  layer_dense(8) %>%
  layer_dense(8) %>%
  layer_dense(8) %>%
  layer_dense(8) %>%
  # output layer
  layer_dense(5, activation = "softmax")

model%>% compile(
  optimizer = optimizer_adam(learning_rate),
  # Loss function to minimize
  loss = loss_sparse_categorical_crossentropy(),
  # List of metrics to monitor
  metrics = list(metric_sparse_categorical_accuracy()),
)
model %>% 
  fit(train_x, train_y,
      epochs=20,
      batch_size=64,
      validation_split=0.2)

#Overfitted baseline model:
model_Overfit <- keras_model_sequential(input_shape = c(dim(train_x)[2],
                                                  dim(train_x)[3],
                                                  1)) %>%
    layer_batch_normalization() %>%
    # convolutional layers
    layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    
    layer_flatten() %>%
    # feed-forward network
    layer_dense(32) %>%
    layer_dense(32) %>%
    layer_dense(32) %>%
    layer_dense(32) %>%
    # output layer
    layer_dense(5, activation = "softmax")

model_overfit %>% compile(
  optimizer = optimizer_adam(learning_rate),
  # Loss function to minimize
  loss = loss_sparse_categorical_crossentropy(),
  # List of metrics to monitor
  metrics = list(metric_sparse_categorical_accuracy()),
)
model_overfit %>% 
  fit(train_x, train_y,
      epochs=20,
      batch_size=64,
      validation_split=0.2)


#####Random Search#####
#uses functions from the Randomsearch.R file

#' Function to create a neural netowrk model given values of hyper parameters
#' @param train_x training data
#' @param weight_decay The weight decay parameter for l2_norm
#' @return a neural network model with the corresponding architecture
call_existing_model <- function(train_x, weight_decay) {
    model <- keras_model_sequential(input_shape = c(dim(train_x)[2],
                                                    dim(train_x)[3],
                                                    1)) %>%
      layer_batch_normalization() %>%
      # convolutional layers
      layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                    kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      layer_conv_2d(filters = 32, kernel_size = c(3,3),
                    kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      layer_conv_2d(filters = 32, kernel_size = c(3,3),
                    kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      layer_conv_2d(filters = 32, kernel_size = c(3,3),
                    kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_max_pooling_2d(pool_size = c(2,2)) %>%
      
      layer_flatten() %>%
      # feed-forward network
      layer_dense(32, kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_dense(32, kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_dense(32, kernel_regularizer = regularizer_l2(weight_decay)) %>%
      layer_dense(32, kernel_regularizer = regularizer_l2(weight_decay)) %>%
      # output layer
      layer_dense(5, activation = "softmax")
  
  return(model)
}



source("Randomsearch.R")
results <- RandomSearch(iterations=20, x_train = train_x, y_train = train_y, 
                        batch_size = 64, epochs=20, 
                        validation_split = 0.2, patience=5)

plot(results$best_model$Model_history)

results$best_model$Hyper_parameters

##dropout 
#Overfitted baseline model:
model_dropout <- keras_model_sequential(input_shape = c(dim(train_x)[2],
                                                dim(train_x)[3],
                                                1)) %>%
  layer_batch_normalization() %>%
  # convolutional layers
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.2) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),
                kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  # feed-forward network
  layer_dense(32, kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_dense(32, kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_dense(32, kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  layer_dense(32, kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dropout(0.2) %>%
  # output layer
  layer_dense(5, activation = "softmax")

model_dropout %>% compile(
  optimizer = optimizer_adam(learning_rate=0.001),
  # Loss function to minimize
  loss = loss_sparse_categorical_crossentropy(),
  # List of metrics to monitor
  metrics = list(metric_sparse_categorical_accuracy()),
)

model_dropout_history <-model_dropout %>% 
  fit(train_x, train_y,
      epochs=20,
      batch_size=64,
      validation_split=0.2)

plot(model_dropout_history, theme_bw = TRUE)



#predict with best model


# predicting on test set
preds <- predict(model_dropout, test_x)
# reducing by 1 since keras encodes (0-9) while R (1-10)
pred_y <- apply(preds, MARGIN = 1, FUN = which.max) - 1

# computing proportion correctly predicted in test set
mean(test_y == pred_y)


library(caret)
conf_matrix <- caret::confusionMatrix(factor(pred_y), factor(test_y))
conf_matrix

class_names <- names(table(spectogram_data$test_genus))
colnames(conf_matrix$table) <- class_names
rownames(conf_matrix$table) <- class_names
rownames(conf_matrix$byClass) <- class_names
conf_matrix
library(xtable)
conf_matrix$table %>% 
  xtable(caption = "Confusion matrix for the best model", 
         label = "tab:confusion_matrix") %>% 
  print(table.position="H", size="\\tiny")

