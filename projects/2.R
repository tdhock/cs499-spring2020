library(data.table)
library(ggplot2)
if(!file.exists("spam.data")){
  download.file("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data", "spam.data")
}

spam.dt <- data.table::fread("spam.data")[sample(.N, 1000)]
N.obs <- nrow(spam.dt)
X.raw <- as.matrix(spam.dt[, -ncol(spam.dt), with=FALSE])
y.vec <- spam.dt[[ncol(spam.dt)]]
X.sc <- scale(X.raw)

set.seed(109)
n.folds <- 10
fold.vec <- sample(rep(1:n.folds, l=nrow(X.sc)))

err.dt.list <- list()
for(validation.fold in 1:n.folds){
  is.validation <- fold.vec == validation.fold
  is.train <- !is.validation
  X.train <- X.sc[is.train, ]
  y.train <- y.vec[is.train]
  for(neighbors in 1:20){
    pred <- class::knn(X.train, X.sc, y.train, k=neighbors)
    pred.y <- as.integer(paste(pred))
    pred.dt <- data.table(
      set=ifelse(is.train, "train", "validation"),
      pred.y,
      label.y=y.vec,
      is.error=pred.y != y.vec)
    mean.err <- pred.dt[, .(
      percent.error=100*mean(is.error)
    ), by=set]
    err.dt.list[[paste(validation.fold, neighbors)]] <- data.table(
      validation.fold, neighbors, mean.err)
  }
}
err.dt <- do.call(rbind, err.dt.list)

ggplot()+
  geom_line(aes(
    neighbors, percent.error, color=set, group=paste(set, validation.fold)),
    data=err.dt)

mean.dt <- err.dt[, .(
  mean.percent=mean(percent.error),
  sd=sd(percent.error)
), by=.(set, neighbors)]
min.dt <- mean.dt[set=="validation"][which.min(mean.percent)]
gg <- ggplot()+
  geom_ribbon(aes(
    neighbors, ymin=mean.percent-sd, ymax=mean.percent+sd, fill=set),
    alpha=0.5,
    data=mean.dt)+
  geom_line(aes(
    neighbors, mean.percent, color=set),
    data=mean.dt)+
  geom_point(aes(
    neighbors, mean.percent, color=set),
    data=min.dt)+
  coord_cartesian(xlim=c(0, 25))
directlabels::direct.label(gg, "last.polygons")
