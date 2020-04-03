library(keras)
## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")

## keras tutorial.
mnist <- dataset_mnist()
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(10, activation = "softmax")

model %>%
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

model %>%
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

predictions <- predict(model, mnist$test$x)
head(predictions, 2)
head(mnist$test$y, 2)
colnames(predictions) <- 0:9

model %>%
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

## SPAM example.
if(!file.exists("spam.data")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data",
    "spam.data")
}

spam.dt <- data.table::fread("spam.data")
label.col <- ncol(spam.dt)
y <- array(spam.dt[[label.col]], nrow(spam.dt))

set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(spam.dt)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test

X.sc <- scale(spam.dt[, -label.col, with=FALSE])
X.train.mat <- X.sc[is.train,]
X.test.mat <- X.sc[is.test,]
X.train.a <- array(X.train.mat, dim(X.train.mat))
X.test.a <- array(X.test.mat, dim(X.test.mat))

y.train <- y[is.train]
y.test <- y[is.test]


n.splits <- 2
split.metrics.list <- list()
for(split.i in 1:n.splits){
  model <- keras_model_sequential() %>%
    layer_dense(input_shape=ncol(X.train.mat), units = 100, activation = "sigmoid", use_bias=FALSE) %>% #hidden layer
    layer_dense(1, activation = "sigmoid", use_bias=FALSE) #output layer
  model %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy" #TODO exact AUC.
    )
  result <- model %>%
    fit(
      x = X.train.mat, y = y.train,
      epochs = 100,
      validation_split = 0.3,
      verbose = 2
    )
  print(plot(result))
  metrics.wide <- do.call(data.table::data.table, result$metrics)
  metrics.wide[, epoch := 1:.N]
  split.metrics.list[[split.i]] <- data.table::data.table(
    split.i, metrics.wide)
}
split.metrics <- do.call(rbind, split.metrics.list)

split.means <- split.metrics[, .(
  mean.val.loss=mean(val_loss),
  sd.val.loss=sd(val_loss)
), by=epoch]
min.dt <- split.means[which.min(mean.val.loss)]
min.dt[, point := "min"]
library(ggplot2)
ggplot()+
  geom_ribbon(aes(
    x=epoch, ymin=mean.val.loss-sd.val.loss, ymax=mean.val.loss+sd.val.loss),
    alpha=0.5,
    data=split.means)+
  geom_point(aes(
    x=epoch, y=mean.val.loss),
    data=split.means)+
  geom_point(aes(
    x=epoch, y=mean.val.loss, color=point),
    data=min.dt)

model <- keras_model_sequential() %>%
  layer_flatten(input_shape = ncol(X.train.mat)) %>% #input layer
  layer_dense(units = 100, activation = "sigmoid", use_bias=FALSE) %>% #hidden layer
  layer_dense(1, activation = "sigmoid", use_bias=FALSE) #output layer
model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy" #TODO exact AUC.
  )
result <- model %>%
  fit(
    x = X.train.mat, y = y.train,
    epochs = min.dt$epoch,
    validation_split = 0,
    verbose = 2
  )
print(plot(result))

model %>%
  evaluate(X.test.mat, y.test, verbose = 0)

y.tab <- table(y.train)
y.baseline <- as.integer(names(y.tab[which.max(y.tab)]))
mean(y.test == y.baseline)
