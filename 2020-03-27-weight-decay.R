library(keras)

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
n.folds <- 5
fold.vec <- sample(rep(1:n.folds, l=nrow(spam.dt)))

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

weight.validation.list <- list()

future::plan("multiprocess")

(weight.decay.vec <- 10^seq(-9, 3, by=0.5))
(weight.decay.new <- weight.decay.vec[
  ! weight.decay.vec %in% names(weight.validation.list)])
weight.validation.new.list <- future.apply::future_lapply(
  weight.decay.new, function(weight.decay){
    ## weight.validation.new.list <- list()
    ## for(weight.decay in weight.decay.new){
    print(weight.decay)
    model <- keras_model_sequential() %>%
      layer_flatten(input_shape = ncol(X.train.mat)) %>% #input layer
      layer_dense(units = 100, activation = "sigmoid", use_bias=FALSE) %>% #hidden layer
      layer_activity_regularization(l2=weight.decay) %>%
      layer_dense(1, activation = "sigmoid", use_bias=FALSE) #output layer
    model %>%
      compile(
        loss = "binary_crossentropy",
        optimizer = "adam",
        metrics = "accuracy" #TODO exact AUC.
      )
    result <- model %>%
      fit(
        x = X.train.mat, y = y.train,
        epochs = 100,
        validation_split = 0.3,
        verbose = 1
      )
    result.epochs <- do.call(data.table::data.table, result$metrics)
    ## weight.validation.new.list[[paste(weight.decay)]] <- data.table::data.table(
    ##   weight.decay, result.epochs[.N])
    data.table::data.table(weight.decay, result.epochs[.N])
  })

weight.validation.list[paste(weight.decay.new)] <- weight.validation.new.list

weight.validation <- do.call(rbind, weight.validation.list)

library(ggplot2)
weight.validation.tall <- nc::capture_melt_single(
  weight.validation,
  set=".*",
  metric="accuracy|loss")
weight.validation.tall[, Set := ifelse(set=="val_", "validation", "subtrain")]
weight.validation.acc <- weight.validation.tall[metric=="accuracy"]
weight.validation.err <- weight.validation.acc[, .(
  weight.decay, set, metric="error", value=1-value, Set)]
weight.validation.show <- rbind(
  weight.validation.tall[metric=="loss"],
  weight.validation.err)
gg <- ggplot()+
  geom_point(aes(
    x=weight.decay, y=value, color=Set),
    data=weight.validation.show)+
  geom_line(aes(
    x=weight.decay, y=value, color=Set),
    data=weight.validation.show)+
  facet_grid(metric ~ ., scales="free")+
  scale_x_log10(limits=c(min(weight.decay.vec), max(weight.decay.vec)*10))+
  scale_y_log10()
directlabels::direct.label(gg, "last.polygons")
