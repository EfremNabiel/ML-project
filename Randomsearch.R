###Note in order to run the following code. It is required that the user defines an own function  call_existing_model(train_x, weight_decay) which specifies the model architecture.
# By default it will assume that weight_decay is  parameter for some kernel_regularizer()

#####Random Search#####

##
#specify call_existing_model() in each script

# example function
#call_existing_model <- function(train_x, weight_decay) {
#  model <- specify keras neural net
#  
#  return(model)
#}


#' Function to sample hyperparameters and creates a model using call_existing_model()
#' and compiles it using the specified optimizer and loss function
#' @param x_train training data
#' @return a compiled model with additional attributes for the hyperparameters

build_model <- function(train_x){
  lr_log <- runif(n=1, max=-1, min=-5)
  weight_decay_log <-  runif(n=1, max=-1, min=-5) 
  lr <- 10^lr_log
  weight_decay <- 10^weight_decay_log
  
  model <- call_existing_model(train_x,weight_decay = weight_decay)
  model %>% compile(
    optimizer = optimizer_adam(learning_rate=lr),
    # Loss function to minimize
    loss = loss_sparse_categorical_crossentropy(),
    # List of metrics to monitor
    metrics = list(metric_sparse_categorical_accuracy()),
  ) 
  attributes(model)$Learning_rate <- lr
  attributes(model)$weight_decay <- weight_decay
  return(model)
}

#' Function to return the model hyperparameters from attributes
hyperparameters_summary <- function(model){
  model_params <- attributes(model)[3:length(attributes(model))]
  #first 2 attributes are keras specific attributes. rest are the hyper parameters
  return(model_params)
}

#' Wrapper function to fit a compiled model
fit_model <- function(model, x_train, y_train, epochs, batch_size,
                      validation_split,...){
  model %>% 
    fit(x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        ...)
}


#' Function to print the hyper parameters of a model
#' @param hp_summary Returned object from the function hyperparameters_summary()
RS.print_hyperparameters <- function(hp_summary){
  for(i in 1:length(hp_summary)){
    attri <- names(hp_summary)[i]
    cat("Current", attri, "is:", 
        ifelse(attri=="Kernel_size", 
               paste0("(", hp_summary[[i]],", ", hp_summary[[i]], ")" ), 
               round(hp_summary[[i]], digits=4)),
        fill=TRUE )
  }
  cat("\n")
}



#' function to return the best model_history results 
#' This is based on the validation accuracy
#' @param model_history the model history object returned from the fit_model() function
RS.get_model_results <- function(model_history){
  metrics <- model_history$metrics
  acc_metric <- metrics[[4]] #asssuming validation accuracy is always at index 4 
  
  index <- which.max(acc_metric)
  best_metrics <- lapply(metrics, FUN=function(metric){metric[[index]]})
  best_metrics$epochs_used <- index
  return(best_metrics)
}


#' Function to create a list object for each iteration of a new model in the RandomSearch 
#' @param iter The current iteration number
#' @param model The model object
#' @param HP_summary The hyperparameters of the model
#' @param model_history The model history object returned from the fit_model() function
#' @param model_results The model results object returned from the RS.get_model_results()
#' @return A list object containing the model, hyperparameters, model history and model results

RS.create_iter_list <- function(iter, model, HP_summary,
                                model_history, model_results){
  list <- list("Model" = model,
               "Hyper_parameters" = HP_summary,
               "Model_history" = model_history,
               "Model_results" = model_results,
               "Iteration" = iter)
  return(list)
}


#' function to update the current best model in the RandomSearch
#' @param new_iter_res_obj The list object returned from the RS.create_iter_list()
#' @param cur_best The current best model
#' @return The updated current best model

RS.update_current_best <- function(new_iter_res_obj, cur_best){
  # For subsequent iterations, compare validation accuracy
  iter_accuracy <- new_iter_res_obj$Model_results[[4]]
  cur_best_accuracy <- cur_best$Model_results[[4]]
  
  if (iter_accuracy > cur_best_accuracy) {
    # Update the current best if the new iteration has a higher validation accuracy
    cur_best <- new_iter_res_obj
  }
  
  return(cur_best)
}

#' function to print the current best model in the RandomSearch, its
#'  hyperparameters and validation results
RS.print_current_best <- function(iter_res_obj){
  parameter_names <- names(iter_res_obj$Hyper_parameters)
  parameter_vals <- iter_res_obj$Hyper_parameters
  validation_names <- names(iter_res_obj$Model_results[3:4])
  validation_res <- iter_res_obj$Model_results[3:4]
  
  cat("\n")
  cat("Current best model is model no.", paste0("<", iter_res_obj$Iteration, ">"), 
      " with the following hyperparameters: \n")
  for(i in 1:length(parameter_names)){
    cat(paste0(parameter_names[i], ":"), ifelse(parameter_names[i]=="Kernel_size", 
                                                paste0("(", parameter_vals[[i]],", ", parameter_vals[[i]], ")" ),
                                                round(parameter_vals[[i]], digits=4)), fill=TRUE)
  }
  cat("\n")
  
  cat("Validation results: ", "\n")
  
  for (i in 1:length(validation_names)){
    cat(paste0(validation_names[i], ":"), round(validation_res[[i]], digits=4),
        fill=TRUE)
  }
  cat("\n")
}

#' function to print that a new model is being tested
RS.print_new_iter <- function(iter){
  emojis <- strrep("\U0001F525", 3)
  line_breaks <- strrep("*", 25)
  cat(line_breaks, "\n", emojis, "New Model!", emojis, "\n")
  cat(line_breaks, "\n")
  cat("Current model number is:", iter, "\n")
}

#' function to print the total execution time of the RandomSearch
RS.print_time <- function(start_time){
  end_time <- Sys.time()
  total_duration <- end_time - start_time
  cat("Total execution time: ", format(total_duration, units = "secs"), "\n")
}

#' main function to implement the RandomSearch method
#' @param iterations the number of iterations to run the RandomSearch. 
#' That is, how many different models to test
#' @param x_train training data
#' @param y_train training labels
#' @param epochs number of epochs to train each model
#' @param batch_size batch size to use for training
#' @param validation_split the proportion of the training data to use for validation
#' @param patience the number of epochs to wait before early stopping
#' @param ... optional arguments to pass to the fit_model() function
#' @return a list object containing all the models and the best model
RandomSearch <- function(iterations, x_train, y_train, epochs,
                         batch_size, validation_split, patience, ...){
  start_time <- Sys.time()
  all_models <- list()
  current_best <- list()
  iteration_index <- 0
  for (i in 1:iterations){
    iteration_index <- iteration_index + 1
    model <- build_model(train_x = x_train)
    #print model summary
    hyper_summary <- hyperparameters_summary(model)
    RS.print_new_iter(iter=iteration_index)
    RS.print_hyperparameters(hyper_summary)
    model_history <- fit_model(model,  x_train, y_train, epochs=epochs,
                               batch_size=batch_size, 
                               validation_split=validation_split, 
                               callbacks=list(callback_early_stopping(
                                 monitor="val_sparse_categorical_accuracy",
                                 patience=patience)),...)
    model_results <- RS.get_model_results(model_history)
    iteration_results <- RS.create_iter_list(iteration_index, model, hyper_summary,
                                             model_history, model_results)
    
    all_models[[i]] <- iteration_results
    
    if(iteration_index==1) {
      cur_best <- iteration_results}
    else {
      cur_best <- RS.update_current_best(iteration_results, cur_best)
    }
    RS.print_current_best(cur_best)
    
  }
  all_results <- list(
    "all_models" = all_models,
    "best_model" = cur_best
  )
  RS.print_time(start_time)
  return(all_results)
}
