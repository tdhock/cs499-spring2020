library(keras)
## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")
future::plan("multiprocess")

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
y.train <- y[is.train]
y.test <- y[is.test]

set.seed(1)
is.subtrain <- sample(rep(c(TRUE, FALSE), l=length(y.train)))
is.validation <- !is.subtrain
X.subtrain.mat <- X.train.mat[is.subtrain,]
X.validation.mat <- X.train.mat[is.validation,]
y.subtrain <- y.train[is.subtrain]
y.validation <- y.train[is.validation]

## analysis of number of hidden units.

units.metrics.list <- list()

(hidden.units.vec <- 2^seq(1, 10))
hidden.units.new <- hidden.units.vec[
  ! hidden.units.vec %in% names(units.metrics.list)]

##for(hidden.units.i in seq_along(hidden.units.new)){
units.metrics.list.new <- future.apply::future_lapply(
  hidden.units.new, function(hidden.units){
    print(hidden.units)
    model <- keras_model_sequential() %>%
      layer_dense(#hidden layer
        input_shape=ncol(X.train.mat),
        units = hidden.units,
        activation = "sigmoid",
        use_bias=FALSE) %>% 
      layer_dense(1, activation = "sigmoid", use_bias=FALSE) #output layer
    model %>%
      compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_adam(lr=0.01),
        metrics = "accuracy" #TODO exact AUC.
      )
    result <- model %>%
      fit(
        x = X.subtrain.mat, y = y.subtrain,
        epochs = 100,
        ##validation_split = 0.3,
        validation_data=list(X.validation.mat, y.validation),
        verbose = 2
      )
    metrics.wide <- do.call(data.table::data.table, result$metrics)
    metrics.wide[, epoch := 1:.N]
    ## units.metrics.list[[paste(hidden.units)]] <- data.table::data.table(
    ##   hidden.units, metrics.wide)
    data.table::data.table(hidden.units, metrics.wide)
  })
units.metrics.list[paste(hidden.units.new)] <- units.metrics.list.new

units.metrics <- do.call(rbind, units.metrics.list)

library(ggplot2)
ggplot()+
  geom_line(aes(
    x=epoch, y=val_loss, color=log10(hidden.units), group=hidden.units),
    data=units.metrics)+
  scale_color_gradient(low="grey", high="red")+
  theme_bw()

ggplot()+
  geom_line(aes(
    x=epoch, y=val_loss),
    data=units.metrics)+
  theme_bw()+
  facet_wrap("hidden.units")

(units.metrics.tall <- nc::capture_melt_single(
  units.metrics,
  set="val_|",
  metric="loss|acc"))
units.metrics.tall[, Set := ifelse(set=="val_", "validation", "subtrain")]

units.loss.tall <- units.metrics.tall[metric=="loss"]
units.loss.min <- units.loss.tall[, .SD[
  which.min(value)], by=.(Set, hidden.units)]
ggplot()+
  geom_line(aes(
    x=epoch, y=value, color=Set),
    data=units.loss.tall)+
  geom_point(aes(
    x=epoch, y=value, color=Set),
    data=units.loss.min)+
  theme_bw()+
  facet_wrap("hidden.units")

ggplot()+
  geom_line(aes(
    x=hidden.units, y=value, color=Set),
    data=units.loss.min)+
  scale_x_log10()


## analyzing number of hidden layers.
layers.metrics.list <- list()

(hidden.layers.vec <- 0:5)
hidden.layers.new <- hidden.layers.vec[
  ! hidden.layers.vec %in% names(layers.metrics.list)]

##for(hidden.layers.i in seq_along(hidden.layers.new)){
layers.metrics.list.new <- future.apply::future_lapply(
  hidden.layers.new, function(hidden.layers){
    print(hidden.layers)
    model <- keras_model_sequential()
    for(layer.i in (0:hidden.layers)[-1]){
      layer_dense(
        model, #hidden layer
        input_shape=ncol(X.train.mat),
        units = 10,
        activation = "relu",
        use_bias=FALSE)
    }
    layer_dense(
      model,
      units=1,
      input_shape=ncol(X.train.mat),
      activation = "sigmoid", use_bias=FALSE) #output layer
    model %>%
      compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_adam(lr=0.001),
        metrics = "accuracy" #TODO exact AUC.
      )
    result <- model %>%
      fit(
        x = X.subtrain.mat, y = y.subtrain,
        epochs = 100,
        ##validation_split = 0.3,
        validation_data=list(X.validation.mat, y.validation),
        verbose = 2
      )
    metrics.wide <- do.call(data.table::data.table, result$metrics)
    metrics.wide[, epoch := 1:.N]
    ## layers.metrics.list[[paste(hidden.layers)]] <- data.table::data.table(
    ##   hidden.layers, metrics.wide)
    data.table::data.table(hidden.layers, metrics.wide)
  })
layers.metrics.list[paste(hidden.layers.new)] <- layers.metrics.list.new

layers.metrics <- do.call(rbind, layers.metrics.list)

(layers.metrics.tall <- nc::capture_melt_single(
  layers.metrics,
  set="val_|",
  metric="loss|acc"))
layers.metrics.tall[, Set := ifelse(set=="val_", "validation", "subtrain")]

layers.loss.tall <- layers.metrics.tall[metric=="loss"]
layers.loss.min <- layers.loss.tall[, .SD[
  which.min(value)], by=.(Set, hidden.layers)]
library(ggplot2)
ggplot()+
  geom_line(aes(
    x=epoch, y=value, color=Set),
    data=layers.loss.tall)+
  geom_point(aes(
    x=epoch, y=value, color=Set),
    data=layers.loss.min)+
  theme_bw()+
  facet_wrap("hidden.layers")

gg <- ggplot()+
  geom_line(aes(
    hidden.layers, value, color=Set),
    data=layers.loss.min)+
  facet_grid(metric ~ .)
directlabels::direct.label(gg, "last.polygons")+
  xlim(0, 6)
