setwd("~/Documents/github/prudential-kaggle/submissions")

# H2O Starter Script
library(h2o)
library(readr)
library(psych)
library(ggplot2)
library(RColorBrewer)
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))
r <- rf(32)
                    
h2o.init(nthreads=-1)

response_count <- 8
weight_matrix <- matrix( rep(0, response_count**2), ncol = response_count, nrow = response_count )
for( i in 1:response_count ) {
  for( j in 1:response_count) {
    if( i == j ) { value = 0 }
    else { value = ((i-j)**2 / (response_count - 1)**2 ) }
    weight_matrix[i, j] <- value
  }
}

categoricalVariables = c("Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7", 
                         "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", 
                         "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", 
                         "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", 
                         "Insurance_History_8", "Insurance_History_9", 
                         "Family_Hist_1", "Medical_History_2", 
                         "Medical_History_3", "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", 
                         "Medical_History_8", "Medical_History_9", "Medical_History_10", "Medical_History_11", "Medical_History_12", 
                         "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", 
                         "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", 
                         "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                         "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
                         "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41")
specifiedTypes = rep("Enum", length(categoricalVariables))
col.types=list(by.col.name=categoricalVariables,types=specifiedTypes)

cat("reading the train and test data\n")
train <- h2o.uploadFile("../input/train.csv", col.types=col.types)
test  <- h2o.uploadFile("../input/test.csv", col.types=col.types)

independentVariables = names(train)[2:(ncol(train)-1)]
dependentVariable = names(train)[128]

cat("creating model")
gbm_hyper_params <- list(ntrees=c(300,500),
                         learn_rate=c(0.03,0.01),
                         max_depth=c(5,10,20),
                         min_rows=c(10,5,3))
frames <- h2o.splitFrame(train, ratios=0.7, seed=1)
training_frame <- frames[[1]]
validation_frame <- frames[[2]]
gbm_grid <- h2o.grid("gbm", grid_id="grid_1", 
                     x=independentVariables, y=dependentVariable, 
                     training_frame = training_frame, validation_frame = validation_frame, 
                     distribution="gaussian", keep_cross_validation_predictions = FALSE,
                     hyper_params = gbm_hyper_params)

show(gbm_grid)

grid_models <- lapply(gbm_grid@model_ids, function(model_id) { model = h2o.getModel(model_id) })

best_kappa <- 9999
best_mse <- 99999
for (i in 1:length(grid_models)) {
  mse <- grid_models[[i]]@model$validation_metrics@metrics$MSE
  if( mse < best_mse ) {
    best_mse <- mse
    best_mse_model <- grid_models[[i]]
  }
  # kappa calc
  predictions <- as.data.frame( predict(grid_models[[i]], validation_frame) )
  actuals <- as.data.frame( validation_frame$Response )
  combined <- cbind(round(predictions$predict), actuals )
  names(combined) <- c("Predicted", "Actual")
  combined[combined$Predicted<1, "Predicted"] <- 1
  combined[combined$Predicted>8, "Predicted"] <- 8

  kappa <- wkappa( combined, w=weight_matrix )
  if( kappa$weighted.kappa < best_kappa ) {
    best_kappa <- kappa$weighted.kappa
    best_kappa_model <- grid_models[[i]]
  }
   
}
best_mse_model
best_kappa_model


 

predictions <- as.data.frame( predict(best_kappa_model, validation_frame) )
actuals <- as.data.frame( validation_frame$Response )
combined <- cbind(round(predictions$predict), actuals )
names(combined) <- c("Predicted", "Actual")
combined[combined$Predicted<1, "Predicted"] <- 1
combined[combined$Predicted>8, "Predicted"] <- 8
# combined[combined$Predicted==3, "Predicted"] <- 2

ggplot(combined, aes(Actual, Predicted)) + geom_jitter()

final_gbm_model <- h2o.gbm(x=independentVariables, y=dependentVariable, training_frame = train,
                           ntrees=400, max_depth=10)

cat("make predictions")
prediction <- as.data.frame( predict(final_gbm_model, test) )
submission <- as.data.frame(test$Id)
submission <- cbind(submission, round(prediction$predict))
names(submission) <- c("Id", "Response")

submission[submission$Response<1, "Response"] <- 1
submission[submission$Response>8, "Response"] <- 8

# without this line the score is 0.58903
# with this line the score is 0.59656
submission[submission$Response==3,"Response"] <- 2

cat("saving the submission file\n")
write_csv(submission, "h2ogbm-6-12-v2.csv")

