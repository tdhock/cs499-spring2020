library(animint2)
library(data.table)

f <- function(x)x^2
min.x <- -5
max.x <- 5
x.vec <- seq(min.x, max.x, l=501)
x.dt <- data.table(x=x.vec)
f.dt <- data.table(x.dt, pred.y=f(x.vec), fun="true")
ggplot()+
  geom_line(aes(
    x, pred.y, color=fun),
    data=f.dt)

generate.data <- function(N.data){
  x <- runif(N.data, min.x, max.x)
  y <- f(x) + rnorm(N.data)
  data.table(x, y)
}
set.seed(1)
N.train <- 10
(train.dt <- generate.data(N.train))
ggplot()+
  geom_point(aes(
    x, y),
    data=train.dt)

set.list <- list(
  train=train.dt,
  test=generate.data(100))

degree.vec <- 1:(N.train-1)
model.dt.list <- list()
error.dt.list <- list()
for(degree in degree.vec){
  right.side.vec <- paste0("I(x^", 1:degree, ")")
  right.side.str <- paste(right.side.vec, collapse="+")
  model.str <- paste("y ~", right.side.str)
  model.formula <- as.formula(model.str)
  model.fit <- lm(model.formula, train.dt)
  model.dt.list[[paste(degree)]] <- data.table(
    x=x.vec,
    pred.y=predict(model.fit, x.dt),
    degree,
    fun=paste0("lm.degree=", degree))
  for(set in names(set.list)){
    set.dt <- set.list[[set]]
    pred.y <- predict(model.fit, set.dt)
    squared.error <- (set.dt$y - pred.y)^2
    error.dt.list[[paste(degree, set)]] <- data.table(
      mse=mean(squared.error),
      degree,
      set)
  }
}
model.dt <- do.call(rbind, model.dt.list)
error.dt <- do.call(rbind, error.dt.list)

all.sets <- data.table(set=names(set.list))[, set.list[[set]], by=set]
best.err <- error.dt[set=="test"][mse==min(mse)]
set.colors <- c(
  train="black",
  test="red")
model.color <- "blue"
viz <- animint(
  funs=ggplot()+
    scale_color_manual(values=set.colors)+
    geom_point(aes(
      x, y, color=set),
      shape=1,
      data=all.sets)+
    geom_line(aes(
      x, pred.y),
      data=model.dt[min(all.sets$y) < pred.y & pred.y < max(all.sets$y)],
      color=model.color,
      showSelected="degree"),
  error=ggplot()+
    scale_color_manual(values=set.colors)+
    geom_line(aes(
      degree, log10(mse), color=set, group=set),
      data=error.dt)+
    geom_point(aes(
      degree, log10(mse), color=set),
      shape=1,
      data=best.err)+
    make_tallrect(error.dt, "degree", color="blue"))
animint2dir(viz, "figure-quadratic-interactive")
if(FALSE){
  animint2gist(viz)
}
