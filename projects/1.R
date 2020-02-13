library(data.table)
library(ggplot2)

## define variables which are specific to each data set.
prefix <- "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
data.list <- list(
  "spam.data"=list(
    step.size=0.5,
    label.fun=function(dt)ncol(dt)),
  "SAheart.data"=list(
    step.size=0.1,
    label.fun=function(dt)ncol(dt)),
  "zip.train.gz"=list(
    step.size=0.9,
    ignore=function(y)! y %in% c(0,1),
    label.fun=function(dt)1))

## code to run on each data set.
for(f in names(data.list)){
  if(!file.exists(f)){
    u <- paste0(prefix, f)
    download.file(u, f)
  }
  full.dt <- data.table::fread(f)
  data.info <- data.list[[f]]
  label.col.id <- data.info$label.fun(full.dt)
  print(label.col.id)
  X.raw <- as.matrix(full.dt[, -label.col.id, with=FALSE])
  y.vec <- full.dt[[label.col.id]]
  is.01 <- y.vec == 0 | y.vec == 1
  X.01 <- X.raw[is.01, ]
  y.01 <- y.vec[is.01]
}

## Read spam data and scale.
spam.dt <- data.table::fread("spam.data")
N.obs <- nrow(spam.dt)
X.raw <- as.matrix(spam.dt[, -ncol(spam.dt), with=FALSE])
y.vec <- spam.dt[[ncol(spam.dt)]]
yt.vec <- ifelse(y.vec==1, 1, -1)
X.sc <- scale(X.raw)

## Compute random assignment into train/validation/test sets, and
## count of each.
set.seed(1) # set random number generator for reproducibility.
prop.vec <- (1:N.obs)/N.obs
set.vec <- sample(ifelse(
  prop.vec < 0.6, "train", ifelse(
    prop.vec < 0.8, "validation", "test")))
head(set.vec)
table(set.vec)/N.obs
(count.tab <- table(set.vec, y.vec))
count.train <- count.tab["train",]
(most.freq.yt <- ifelse(names(count.train)[which.max(count.train)]==1, 1, -1))

computeGradient <- function(X, y.tilde, weightVector){
  pred.vec <- X %*% weightVector # theta^T x(i)
  in.exp <- y.tilde * pred.vec
  denominator <- as.numeric(1+exp(in.exp))
  colMeans(-as.numeric(y.tilde) * X / denominator)
}

(grad.mat <- matrix(NA, 5, 2))
for(i in 1:nrow(X)){
  x.vec <- X[i,]
  pred <- t(weightVector) %*% x.vec
  denom <- 1+exp(y.tilde[i]*pred)
  grad.mat[i,] <- -y.tilde[i]*x.vec/denom
}

GradientDescent <- function(X, y.tilde, step.size, max.iterations){
  weightVector <- rep(0, ncol(X))
  weightMatrix <- matrix(0, ncol(X), max.iterations)
  for(i in 1:max.iterations){
    grad.vec <- computeGradient(X, y.tilde, weightVector)
    weightVector <- weightVector - step.size * grad.vec
    weightMatrix[, i] <- weightVector
  }
  weightMatrix
}

is.train <- set.vec == "train"
w.mat <- GradientDescent(X.sc[is.train,], yt.vec[is.train], 0.2, 5000)

pred.mat <- X.sc %*% w.mat
metric.list <- list(
  mean.log.loss=function(score.mat, y.tilde){
    colMeans(log(1+exp(-y.tilde * score.mat)))
  },
  error.percent=function(score.mat, y.tilde){
    score.mat <- ifelse(score.mat>0, 1, -1)
    100*colMeans(y.tilde != score.mat)
  })

metric.dt.list <- list()
for(set.name in c("train", "validation")){
  is.set <- set.vec == set.name
  set.pred.mat <- pred.mat[is.set,]
  set.yt.vec <- yt.vec[is.set]
  for(metric.name in names(metric.list)){
    metric.fun <- metric.list[[metric.name]]
    metric.vec <- metric.fun(set.pred.mat, set.yt.vec)
    metric.dt.list[[paste(set.name, metric.name)]] <- data.table(
      set.name, metric.name, value=metric.vec, iteration=seq_along(metric.vec))
  }
}
metric.dt <- do.call(rbind, metric.dt.list)
min.dt <- metric.dt[, data.table(
  what="min",
  .SD[which.min(value)]
), by=.(set.name, metric.name)]
ggplot()+
  geom_line(aes(
    iteration, value, color=set.name),
    data=metric.dt)+
  geom_point(aes(
    iteration, value, color=set.name, fill=what),
    shape=21,
    data=min.dt)+
  scale_fill_manual(values=c(min="white"))+
  scale_color_manual(values=c(train="black", validation="red"))+
  theme_bw()+
  theme(panel.margin=grid::unit(0, "lines"))+
  facet_grid(metric.name ~ ., scales="free")

best.dt <- min.dt[set.name=="validation" & metric.name=="mean.log.loss"]

best.weight.vec <- w.mat[, best.dt$iteration]
model.list <- list(
  logisticRegression=function(X.mat)X.mat %*% best.weight.vec,
  baseline=function(X.mat)matrix(most.freq.yt, nrow(X.mat)))
err.dt.list <- list()
for(set.name in c("train", "validation", "test")){
  is.set <- set.vec == set.name
  set.X.mat <- X.sc[is.set,]
  set.yt.vec <- yt.vec[is.set]
  for(model.name in names(model.list)){
    model.fun <- model.list[[model.name]]
    set.pred.vec <- model.fun(set.X.mat)
    err.dt.list[[paste(set.name, model.name)]] <- data.table(
      set.name, model.name,
      error.percent=metric.list$error.percent(set.pred.vec, set.yt.vec))
  }
}
err.dt <- do.call(rbind, err.dt.list)
(set.err.dt <- dcast(err.dt, set.name ~ model.name))

roc.dt.list <- list()
pred.point.list <- list()
for(model.name in names(model.list)){
  model.fun <- model.list[[model.name]]
  set.pred.vec <- model.fun(set.X.mat)
  pred.yt.vec <- ifelse(set.pred.vec > 0, 1, -1)
  is.false.positive <- pred.yt.vec==1 & set.yt.vec==-1
  is.negative <- set.yt.vec==-1
  is.true.positive <- pred.yt.vec==1 & set.yt.vec==1
  is.positive <- set.yt.vec==1
  pred.point.list[[model.name]] <- data.table(
    model.name,
    FPR=sum(is.false.positive)/sum(is.negative),
    TPR=sum(is.true.positive)/sum(is.positive))
  wroc.df <- WeightedROC::WeightedROC(set.pred.vec, set.yt.vec)
  roc.dt.list[[model.name]] <- data.table(
    model.name,
    wroc.df)
}
roc.dt <- do.call(rbind, roc.dt.list)
pred.point <- do.call(rbind, pred.point.list)

ggplot()+
  geom_path(aes(
    FPR, TPR, color=model.name),
    data=roc.dt)+
  scale_fill_manual(values=c(predicted="white"))+
  coord_equal()+
  geom_point(aes(
    FPR, TPR, color=model.name, fill=what),
    shape=21,
    data=data.table(what="predicted", pred.point))
