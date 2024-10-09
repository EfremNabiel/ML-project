# Model in keras syntax
library(keras)

# loading data into R
load("./data/rnn_data.RData")

train_x <- rnn_data$train_x
train_y <- rnn_data$train_y
test_x <- rnn_data$test_x
test_y <- rnn_data$test_y



# number of categories to predict
n_categories <- length(unique(train_y))

# reshaping data into shape expected by rnn model
train_x <- array_reshape(train_x, c(nrow(train_x), 1, ncol(train_x)))
test_x <- array_reshape(test_x, c(nrow(test_x), 1, ncol(test_x)))

# defining a baseline model
rnn_model <- keras_model_sequential(input_shape = dim(train_x)[-1]) %>%
  # recurrent net
  layer_lstm(units = 32) %>%
  # feed-forward net
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  # output classification layer
  layer_dense(units = n_categories, activation = "softmax")

# compiling model
rnn_model %>% compile(
  optimizer = "adam",
  # Loss function to minimize
  loss = loss_sparse_categorical_crossentropy(),
  # List of metrics to monitor
  metrics = list(metric_sparse_categorical_accuracy()),
)   

# fitting model
rnn_history <- rnn_model %>% fit(
  train_x,
  train_y,
  batch_size = 64,
  epochs = 20,
  validation_split = 0.2,
  verbose = 1
)

#####Random Search#####

#' Function to create a neural netowrk model given values of hyper parameters
#' @param train_x training data
#' @param weight_decay The weight decay parameter for l2_norm
#' @return a neural network model with the corresponding architecture
call_existing_model <- function(train_x, weight_decay) {
  model <- keras_model_sequential(input_shape = dim(train_x)[-1]) %>%
    # recurrent net
    layer_lstm(units = 32,
               kernel_regularizer = regularizer_l2(weight_decay)) %>%
    # feed-forward net
    layer_dense(units = 32, activation = "relu",
                kernel_regularizer = regularizer_l2(weight_decay)) %>%
    layer_dense(units = 32, activation = "relu",
                kernel_regularizer = regularizer_l2(weight_decay)) %>%
    # output classification layer
    layer_dense(units = 5, activation = "softmax")
  
  return(model)
}

source("Randomsearch.R")

results <- RandomSearch(iterations=20, x_train = train_x, y_train = train_y, 
                        batch_size = 64, epochs=20, 
                        validation_split = 0.2, patience=5)

plot(results$best_model$Model_history)

results$best_model$Hyper_parameters

# best model learning_rate 0.0002396344
# best weight decay 0.0001011969

#### regularizing model ####
final_rnn_model <- keras_model_sequential(input_shape = dim(train_x)[-1]) %>%
  # recurrent net
  layer_lstm(units = 32, kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(0.25) %>%
  # feed-forward net
  layer_dense(units = 32, activation = "relu", kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 32, activation = "relu", kernel_regularizer = regularizer_l2(0.001)) %>%
  # output classification layer
  layer_dense(units = n_categories, activation = "softmax")

# compiling model
final_rnn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0001 ),
  # Loss function to minimize
  loss = loss_sparse_categorical_crossentropy(),
  # List of metrics to monitor
  metrics = list(metric_sparse_categorical_accuracy())
)   

# fitting model
final_rnn_history <- final_rnn_model %>% fit(
  train_x,
  train_y,
  batch_size = 64,
  epochs = 20,
  validation_split = 0.2,
  verbose = 1
)

# evaluating on test set
preds <- predict(final_rnn_model, test_x)

pred_y <- apply(preds, MARGIN = 1, FUN = which.max) - 1

mean(test_y == pred_y)

test_y[which(test_y != pred_y)]
pred_y[which(test_y != pred_y)]

c("Crypturellus", "Nothoprocta", "Nothura", "Ortalis", "Tinamus")

library(caret)
conf_matrix_rnn <- caret::confusionMatrix(factor(pred_y), factor(test_y))


class_names <- names(table(rnn_data$test_genus))
colnames(conf_matrix_rnn$table) <- class_names
rownames(conf_matrix_rnn$table) <- class_names
rownames(conf_matrix_rnn$byClass) <- class_names
conf_matrix_rnn

library(xtable)
conf_matrix_rnn$table %>% 
  xtable(caption = "Confusion matrix for the best model", 
         label = "tab:confusion_matrix") %>% 
  print(table.placement="H", size="\\tiny")
