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
set.seed(2)
N.train <- 10
(train.dt <- generate.data(N.train))
ggplot()+
  geom_point(aes(
    x, y),
    data=train.dt)

set.list <- list(
  test=generate.data(100),
  train=train.dt,
  grid=data.table(x=x.vec, y=NA_real_))
all.sets <- data.table(
  set=names(set.list)
)[
, set.list[[set]], by=set
]

degree.vec <- 0:(N.train-1)
model.dt.list <- list()
error.dt.list <- list()
for(degree in degree.vec){
  pred.y <- if(degree==0){
    mean(train.dt$x)
  }else{
    right.side.vec <- paste0("I(x^", 1:degree, ")")
    right.side.str <- paste(right.side.vec, collapse="+")
    model.str <- paste("y ~", right.side.str)
    model.formula <- as.formula(model.str)
    model.fit <- lm(model.formula, train.dt)
    predict(model.fit, all.sets)
  }
  model.dt.list[[paste(degree)]] <- data.table(
    all.sets,
    pred.y,
    degree,
    fun=paste0("lm.degree=", degree))
}
model.dt <- do.call(rbind, model.dt.list)
error.dt <- model.dt[
set!="grid", .(
  mse=mean((y - pred.y)^2)
), by=.(degree,set)
][
, mse.thresh := ifelse(mse<1e-10, 0, mse)
]

best.err <- error.dt[set=="test"][mse==min(mse)]
set.colors <- c(
  train="black",
  test="red")
model.color <- "blue"
expand <- 3
not.grid <- model.dt[set!="grid"]
model.dt[, pred.thresh := ifelse(
  pred.y < min(not.grid$y)-expand, -Inf,
  ifelse(pred.y > max(not.grid$y)+expand, Inf, pred.y))]
(viz <- animint(
  error=ggplot()+
    ylab("log10(mean squared error)")+
    scale_x_continuous("polynomial degree", breaks=degree.vec)+
    scale_color_manual(values=set.colors)+
    geom_line(aes(
      degree, log10(mse.thresh), color=set, group=set),
      showSelected="set",
      help="Error of the linear model as a function of the degree of the polynomial basis expansion",
      data=error.dt)+
    geom_point(aes(
      degree, log10(mse.thresh), color=set),
      shape=1,
      showSelected="set",
      help="Minimum error on the test set",
      fill="white",
      data=best.err)+
    theme(legend.position="none")+
    make_tallrect(error.dt, "degree", color=NA, fill="blue"),
  funs=ggplot()+
    xlab("input/feature")+
    ylab("output/label")+
    scale_color_manual(values=set.colors)+
    geom_point(aes(
      x, y, color=set),
      shape=1,
      help="Data points to which a linear model is fit (train=black) and evaluated (test=red), using a polynomial basis expansion",
      fill=NA,
      size=4,
      data=not.grid)+
    geom_line(aes(
      x, pred.thresh, key="pred"),
      data=model.dt[set=="grid"],
      color=model.color,
      help="Predicted values from the linear model with the selected basis expansion",
      showSelected="degree"),
  duration=list(degree=1000),
  out.dir="figure-quadratic-interactive",
  title="Overfitting using linear model polynomial degree",
  source="https://github.com/tdhock/cs499-spring2020/blob/master/2020-02-03-capacity/figure-quadratic-interactive.R"))
if(FALSE){
  animint2pages(viz, "2020-02-03-capacity-polynomial-degree", chromote_sleep_seconds = 5)
}
