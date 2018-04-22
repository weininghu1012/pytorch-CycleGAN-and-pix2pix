datagen <- image_data_generator(rescale = 1/255)
batch_size <- 10

extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))  
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}


base_dir <- "/Users/JasmineJu/Desktop"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")


train <- extract_features(train_dir,100)
validation <- extract_features(validation_dir, 20)
batch_size <- 1
test <- extract_features(test_dir, 1)

dim(train$features)

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)


model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 5,
  validation_data = list(validation$features, validation$labels)
)

plot(history)




layer_outputs <- lapply(model$layers[1], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
activations <- activation_model %>% predict(train$features)
activations <- rbind(activations,activation_model %>% predict(test$features))

activations <- (activations - apply(activations,1,sd))/apply(activations,1,sd)


library("flexclust")
cl1 = kcca(activations, k=2, kccaFamily("kmeans"))
image(cl1)
points(activations, col =  c(train$label+1, "4"))

length( c(train$label+1, "4"))
dim(activations)

points(dat[dat[["train"]]==TRUE, 1:2], col=pred_train, pch=19, cex=0.3)
points(dat[dat[["train"]]==FALSE, 1:2], col=pred_test, pch=22, bg="orange")



layer_outputs <- lapply(model$layers[1], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
activations <- activation_model %>% predict(train$features)


