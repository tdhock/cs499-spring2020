library(ggplot2)
library(data.table)

f <- function(x)x^2
min.x <- -5
max.x <- 5
x.vec <- seq(min.x, max.x, l=101)
x.dt <- data.table(x=x.vec)
f.dt <- data.table(x.dt, pred.y=f(x.vec), fun="true")
ggplot()+
  geom_line(aes(
    x=x, y=pred.y, color=fun),
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

gg.funs <- ggplot()+
  geom_point(aes(
    x, y),
    data=train.dt)+
  geom_line(aes(
    x, pred.y, color=fun),
    data=model.dt[degree %in% c(1,2,N.train-1)])+
  coord_cartesian(ylim=range(train.dt$y))
print(gg.funs)

png("figure-quadratic-funs.png", width=6, height=4, units="in", res=100)
print(gg.funs)
dev.off()

best.err <- error.dt[set=="test"][mse==min(mse)]
gg.err <- ggplot(error.dt, aes(degree, log10(mse), color=set))+
  geom_line()+
  geom_point()+
  geom_point(
    color="black",
    shape=1,
    data=best.err)

png("figure-quadratic.png", width=6, height=4, units="in", res=100)
print(gg.err)
dev.off()
