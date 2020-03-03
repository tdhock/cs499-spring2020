library(ggplot2)
library(data.table)
spam.dt <- data.table::fread("spam.data")
label.col.i <- ncol(spam.dt)
X.mat <- as.matrix(spam.dt[, -label.col.i, with=FALSE])
yt.vec <- ifelse(spam.dt[[label.col.i]]==1, 1, -1)
X.sc <- scale(X.mat)

## Define number of hidden units.
architecture <- c(ncol(X.sc), 50, 20, 1)

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
    input, output, fill=weight),
    data=weight.dt)+
  scale_fill_gradient2()+
  coord_equal()

## Pick one observation (SGD)
obs.i <- 1
x <- X.sc[obs.i,]
yt <- yt.vec[obs.i]

## Forward propagation.
h.list <- list(as.numeric(x))
for(layer.i in 1:length(weight.mat.list)){
  a.vec <- weight.mat.list[[layer.i]] %*% h.list[[layer.i]]
  h.list[[layer.i+1]] <- if(layer.i==length(weight.mat.list)){
    a.vec #last layer activation is identity.
  }else{
    1/(1+exp(-a.vec))
  }
}
str(h.list)

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
str(grad.w.list)

