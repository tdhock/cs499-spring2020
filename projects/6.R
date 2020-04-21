library(keras)
## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")
## zip.train data

if(!file.exists("zip.train.gz")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz",
    "zip.train.gz")
}
zip.train <- data.table::fread("zip.train.gz")


a.size <- c(2, 3, 3, 1)
a <- array(1:prod(a.size), a.size)
str(a)

zip.some <- zip.train[1:10]
zip.X.array <- array(
  unlist(zip.train[1:nrow(zip.train),-1]),
  c(nrow(zip.train), 16, 16, 1))
zip.class.tab <- table(zip.train$V1)
zip.y.mat <- keras::to_categorical(zip.train$V1, length(zip.class.tab))
str(zip.y.mat)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = dim(zip.X.array)[-1]) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')
model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

## deep dense model.
model2 <- keras_model_sequential() %>%
  layer_flatten(input_shape = input_shape) %>% 
  layer_dense(units = 270, activation = 'relu') %>% 
  layer_dense(units = 270, activation = 'relu') %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')
model2 %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


zip.some[, observation.i := 1:.N]
zip.some.tall <- zip.some[, {
  data.table::data.table(
    label=V1,
    col.i=rep(1:16, 16),
    row.i=rep(1:16, each=16),
    intensity=as.numeric(.SD[, paste0("V", 2:257), with=FALSE]))
}, by=observation.i]

library(ggplot2)
ggplot()+
  geom_tile(aes(
    x=col.i, y=-row.i, fill=intensity),
    data=zip.some.tall)+
  facet_wrap(observation.i + label ~ ., labeller=label_both)+
  coord_equal()+
  theme_bw()+
  theme(panel.margin=grid::unit(0, "lines"))+
  scale_fill_gradient(low="white", high="black")

## MNIST data 

mnist <- keras::dataset_mnist()

## https://tensorflow.rstudio.com/guide/keras/examples/mnist_cnn/
mnist.some.tall.list <- list()
for(observation.i in 1:10){
  obs.mat <- mnist$train$x[observation.i, ,]
  mnist.some.tall.list[[observation.i]] <- data.table::data.table(
    observation.i,
    row.i=as.integer(row(obs.mat)),
    col.i=as.integer(col(obs.mat)),
    intensity=as.numeric(obs.mat)/255,
    label=mnist$train$y[observation.i])
}
mnist.some.tall <- do.call(rbind, mnist.some.tall.list)

library(ggplot2)
ggplot()+
  geom_tile(aes(
    x=col.i, y=-row.i, fill=intensity),
    data=mnist.some.tall)+
  facet_wrap(observation.i + label ~ ., labeller=label_both)+
  coord_equal()+
  theme_bw()+
  theme(panel.margin=grid::unit(0, "lines"))+
  scale_fill_gradient(low="white", high="black")

# Data Preparation -----------------------------------------------------

batch_size <- 128 # for training in SGD algo.
num_classes <- 10 # different digits.
epochs <- 12 # passes through train data.

# Input image dimensions
img_rows <- 28
img_cols <- 28



# Redefine  dimension of train/test inputs
N <- 1000
# Convert class vectors to binary class matrices
y_train <- to_categorical(mnist$train$y[1:N], num_classes)
y_test <- to_categorical(mnist$test$y[1:N], num_classes)
max.intensity <- 255
x_train <- array_reshape(
  mnist$train$x[1:N, ,] / max.intensity, c(N, img_rows, img_cols, 1))
x_test <- array_reshape(
  mnist$test$x[1:N, ,]/max.intensity, c(N, img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')
# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

## deep dense model.
model <- keras_model_sequential() %>%
  layer_flatten(input_shape = input_shape) %>% 
  layer_dense(units = 700, activation = 'relu') %>% 
  layer_dense(units = 700, activation = 'relu') %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')
model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)


scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)
# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
