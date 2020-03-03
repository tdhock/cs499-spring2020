library(ggplot2)
library(data.table)
spam.dt <- data.table::fread("spam.data")
label.col.i <- ncol(spam.dt)
X.mat <- as.matrix(spam.dt[, -label.col.i, with=FALSE])
yt.vec <- ifelse(spam.dt[[label.col.i]]==1, 1, -1)
X.sc <- scale(X.mat)

n.folds <- 5
set.seed(1)
fold.vec <- sample(rep(1:n.folds, l=length(yt.vec)))

validation.fold <- 1
is.validation <- fold.vec == validation.fold
is.train <- !is.validation
table(is.train)

## also extreme inefficient, avoid!!
for(f in fold.vec){
  if(f==validation.fold){
    validation.data <- c(validation.data, new.data)
  }
}

## Define number of hidden units.
architecture <- c(ncol(X.sc), 20, 1)
## Initialization of weights to random numbers close to zero.
set.seed(1)
weight.mat.list <- list()
for(layer.i in 1:(length(architecture)-1)){
  mat.nrow <- architecture[[layer.i+1]]
  mat.ncol <- architecture[[layer.i]]##u^(l-1)
  weight.mat.list[[layer.i]] <- matrix(
    rnorm(mat.nrow*mat.ncol), mat.nrow, mat.ncol)
}
str(weight.mat.list)

## visualize the weights.
weight.dt.list <- list()
for(layer.i in seq_along(weight.mat.list)){
  w.mat <- weight.mat.list[[layer.i]]
  weight.dt.list[[layer.i]] <- data.table(
    layer.i,
    weight=as.numeric(w.mat),
    output=as.integer(row(w.mat)),
    input=as.integer(col(w.mat)))
}
weight.dt <- do.call(rbind, weight.dt.list)
ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ layer.i, labeller=label_both)+
  geom_tile(aes(
    x=input, y=output, fill=weight),
    data=weight.dt)+
  scale_fill_gradient2()+
  coord_equal()

ForwardProp <- function(input.mat, w.list){
  h.list <- list(input.mat)
  for(layer.i in 1:length(w.list)){
    right.side <- h.list[[layer.i]]
    left.side <- w.list[[layer.i]]
    a.vec <- left.side %*% right.side
    h.list[[layer.i+1]] <- if(layer.i==length(w.list)){
      a.vec #last layer activation is identity.
    }else{
      1/(1+exp(-a.vec))
    }
  }
  h.list
}

epoch <- 0
yt.train <- yt.vec[is.train]
X.train <- X.sc[is.train,]
loss.dt.list <- list()

## Pick one observation (SGD)
epoch <- epoch+1
obs.vec <- sample(seq_along(yt.train))
for(iteration.i in seq_along(obs.vec)){#one pass through the train observations.
  obs.i <- obs.vec[[iteration.i]]
  ##cat(sprintf("%4d / %4d %d\n", iteration.i, length(obs.vec), obs.i))
  x <- X.train[obs.i,]
  yt <- yt.train[obs.i]
  ## Forward propagation.
  h.list <- ForwardProp(x, weight.mat.list)
  ##str(h.list)
  ## Back propagation.
  grad.w.list <- list()
  for(layer.i in length(weight.mat.list):1){
    grad.a <- if(layer.i==length(weight.mat.list)){
      -yt / (1+exp(yt*h.list[[length(h.list)]]))
    }else{
      grad.h <- t(weight.mat.list[[layer.i+1]]) %*% grad.a
      h.vec <- h.list[[layer.i+1]]
      grad.h * h.vec * (1-h.vec)
    }
    grad.w.list[[layer.i]] <- grad.a %*% t(h.list[[layer.i]])
  }
  ##str(grad.w.list)
  ## Take a step in the negative gradient direction.
  step.size <- 0.05
  for(layer.i in seq_along(weight.mat.list)){
    weight.mat.list[[layer.i]] <-
      weight.mat.list[[layer.i]] - step.size * grad.w.list[[layer.i]]
  }
  ##end of iteration.
}##end of epoch.
pred.h.list <- ForwardProp(t(X.sc), weight.mat.list)
pred.vec <- as.numeric(pred.h.list[[length(pred.h.list)]])
LogisticLoss <- function(pred, label){
  log(1+exp(-label*pred))
}
loss.vec <- LogisticLoss(pred.vec, yt.vec)
loss.dt.list[[epoch]] <- data.table(
  epoch,
  loss=loss.vec,
  set=ifelse(is.train, "train", "validation")
)[, .(
  mean.loss=mean(loss)
), by=.(epoch, set)]

## Please avoid this idiom for accumulating data tables (inefficient
## quadratic time/space complexity).
loss.dt <- rbind(loss.dt, new.dt)

## Do this instead (linear time/space).
loss.dt <- do.call(rbind, loss.dt.list)
ggplot()+
  geom_line(aes(
    x=epoch, y=mean.loss, color=set),
    data=loss.dt)
