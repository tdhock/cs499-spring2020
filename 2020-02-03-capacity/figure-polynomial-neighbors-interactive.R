library(animint2)
library(data.table)
pattern.f.seed <- function(pattern, fun, seed){
  data.table(pattern=pattern, fun=list(fun), seed=seed)
}
max.x <- 5
min.x <- -max.x
pattern.dt <- rbind(
  pattern.f.seed("constant", function(x)1, 4),
  pattern.f.seed("cubic", function(x)x^3/(max.x^3), 7),#1?
  pattern.f.seed("quadratic", function(x)x^2/(max.x^2), 11),#6?
  pattern.f.seed("linear", function(x)x/max.x, 3))
grid.x.vec <- seq(min.x, max.x, l=401)
set.seed(7)#4?
N.train <- 10
max.degree <- N.train-1
N.total <- 100
x <- runif(N.total, min.x, max.x)
set <- rep("test", N.total)
set[1:N.train] <- "train"
model.dt.list <- list()
lm.poly.deg <- "linear model poly. deg."
num.nn <- "num. nearest neighbors"
for(pattern.i in 1:nrow(pattern.dt)){
  pattern.row <- pattern.dt[pattern.i]
  set.seed(pattern.row$seed)
  f <- pattern.row$fun[[1]]
  y <- f(x) + rnorm(N.total, sd=0.1)
  all.sets <- rbind(
    data.table(set, x, y),
    data.table(set="grid", x=grid.x.vec, y=NA_real_))
  yrange <- all.sets[set!="grid", range(y)]
  all.sets[, ynorm := (y-yrange[1])/diff(yrange)]
  degree.vec <- 0:max.degree
  train.set <- all.sets[set=="train"]
  for(degree in degree.vec){
    pred.y <- if(degree==0){
      train.set[, mean(ynorm)]
    }else{
      right.side.vec <- paste0("I(x^", 1:degree, ")")
      right.side.str <- paste(right.side.vec, collapse="+")
      model.str <- paste("ynorm ~", right.side.str)
      model.formula <- as.formula(model.str)
      model.fit <- lm(model.formula, train.set)
      predict(model.fit, all.sets)
    }
    model.dt.list[[paste(pattern.i, degree, "lm")]] <- data.table(
      pattern.row, 
      all.sets,
      pred.y,
      parameter=degree,
      regularization=lm.poly.deg)
  }
  for(num.neighbors in 1:N.train){
    kfit <- FNN::knn.reg(
      train.set[, .(x)],
      all.sets[, .(x)],
      train.set$ynorm,
      num.neighbors)
    model.dt.list[[paste(pattern.i, num.neighbors, "nn")]] <- data.table(
      pattern.row, 
      all.sets,
      pred.y=kfit[["pred"]],
      parameter=num.neighbors,
      regularization=num.nn)
  }
}
model.dt <- do.call(rbind, model.dt.list)
error.dt <- model.dt[
  set!="grid", .(
    mse=mean((ynorm - pred.y)^2)
  ), by=.(pattern, regularization, parameter, set)
][
, mse.thresh := ifelse(mse<1e-10, 0, mse)
]
best.err <- error.dt[set=="test"][, .SD[mse==min(mse)], by=.(pattern, regularization)]
set.colors <- c(
  train="black",
  test="red")
reg.dt <- rbind(
  data.table(regularization=lm.poly.deg, color="blue", size=3, hjust=0, fun="max"),
  data.table(regularization=num.nn, color="green", size=2, hjust=1, fun="min"))
model.colors <- reg.dt[, structure(color, names=regularization)]
model.sizes <- reg.dt[, structure(size, names=regularization)]
expand <- 0.1
not.grid <- model.dt[set!="grid"]
model.dt[, pred.thresh := ifelse(
  pred.y < min(not.grid$ynorm)-expand, -Inf,
  ifelse(pred.y > max(not.grid$ynorm)+expand, Inf, pred.y))]
tallrect.dt <- unique(error.dt[, .(regularization, parameter)])
(text.dt <- error.dt[set=="test"][reg.dt, on="regularization"][, {
  FUN <- get(fun)
  .SD[parameter==FUN(parameter)]
}, by=fun])
duration.list <- list(pattern=1000)
for(regularization in names(model.colors)){
  duration.list[[regularization]] <- 1000
}
height.pixels <- 500
(viz <- animint(
  error=ggplot()+
    ggtitle("Select pattern and models")+
    theme(legend.position="none")+
    theme_animint(height=height.pixels)+
    scale_y_continuous("log10(mean squared error)")+
    scale_x_continuous(
      "regularization parameter",
      limits=range(tallrect.dt$parameter)+c(-1,1),
      breaks=unique(tallrect.dt$parameter))+
    scale_color_manual(values=set.colors)+
    scale_fill_manual(values=model.colors)+
    facet_grid(regularization ~ ., scales="free")+
    geom_tallrect(aes(
      xmin=parameter-0.5,
      xmax=parameter+0.5,
      fill=regularization),
      alpha=0.5,
      color=NA,
      data=tallrect.dt,
      showSelected=c("regularization"),
      clickSelects=c(regularization="parameter"))+
    geom_line(aes(
      parameter, log10(mse.thresh), color=set, group=paste(pattern, set)),
      clickSelects="pattern",
      size=5,
      alpha_off=0.1,
      showSelected=c("regularization", "set"),
      data=error.dt)+
    geom_point(aes(
      parameter, log10(mse.thresh), color=set),
      shape=1,
      fill="white",
      alpha_off=0.1,
      size=4,
      clickSelects="pattern",
      showSelected=c("regularization", "set"),
      data=best.err)+
    geom_label_aligned(aes(
      parameter, log10(mse.thresh),
      hjust=hjust,
      label=pattern,
      color=set),
      clickSelects="pattern",
      showSelected=c("regularization", "set"),
      data=text.dt),
  funs=ggplot()+
    ggtitle("Selected pattern (points) and models (curves)")+
    xlab("input/feature")+
    ylab("output/label")+
    theme_animint(height=height.pixels)+
    scale_fill_manual(values=set.colors)+
    scale_color_manual(values=model.colors)+
    scale_size_manual(values=model.sizes)+
    geom_point(aes(
      x, ynorm, fill=set, key=x),
      size=4,
      showSelected=c("set","pattern"),
      data=not.grid)+
    geom_line(aes(
      x, pred.thresh,
      size=regularization,
      key=regularization,
      group=regularization,
      color=regularization),
      data=model.dt[set=="grid"],
      showSelected=c("pattern", regularization="parameter")),
  duration=duration.list,
  out.dir="figure-polynomial-neighbors-interactive",
  title="Overfitting using linear model polynomial degree and nearest neighbors",
  first=structure(list(10), names=num.nn),
  source="https://github.com/tdhock/cs499-spring2020/blob/master/2020-02-03-capacity/figure-polynomial-neighbors-interactive.R"))
if(FALSE){
  animint2pages(viz, "2023-12-04-degree-neighbors")
}

