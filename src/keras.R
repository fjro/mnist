# adapted from https://keras.rstudio.com/

# install_keras("conda")
# get the data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape the 3d arrays into row major 2d arrays

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# encode categories 
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# define a sequential model
model <- keras_model_sequential() 
class(model)

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# train and evaluate the model
history <- model %>% 
  fit(
    x_train, 
    y_train, 
    epochs = 30, 
    batch_size = 128, 
    validation_split = 0.2
)

plot(history)

# look at the data on the test data
model %>% 
  evaluate(x_test, y_test)


model %>% 
  predict_classes(x_test)
