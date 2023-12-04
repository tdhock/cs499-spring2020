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
set.seed(6)
N.train <- 10
max.degree <- N.train-1
N.total <- 100
x <- runif(N.total, min.x, max.x)
set <- rep("test", N.total)
set[1:N.train] <- "train"
model.dt.list <- list()
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
  for(degree in degree.vec){
    pred.y <- if(degree==0){
      all.sets[set=="train", mean(ynorm)]
    }else{
      right.side.vec <- paste0("I(x^", 1:degree, ")")
      right.side.str <- paste(right.side.vec, collapse="+")
      model.str <- paste("ynorm ~", right.side.str)
      model.formula <- as.formula(model.str)
      model.fit <- lm(model.formula, all.sets[set=="train"])
      predict(model.fit, all.sets)
    }
    model.dt.list[[paste(pattern.i, degree)]] <- data.table(
      pattern.row, 
      all.sets,
      pred.y,
      degree,
      fun=paste0("lm.degree=", degree))
  }
}
model.dt <- do.call(rbind, model.dt.list)

error.dt <- model.dt[
  set!="grid", .(
    mse=mean((ynorm - pred.y)^2)
  ), by=.(pattern, degree,set)
][
, mse.thresh := ifelse(mse<1e-10, 0, mse)
]
best.err <- error.dt[set=="test"][, .SD[mse==min(mse)], by=pattern]
set.colors <- c(
  train="black",
  test="red")
model.color <- "blue"
expand <- 0.1
not.grid <- model.dt[set!="grid"]
model.dt[, pred.thresh := ifelse(
  pred.y < min(not.grid$ynorm)-expand, -Inf,
  ifelse(pred.y > max(not.grid$ynorm)+expand, Inf, pred.y))]
(viz <- animint(
  error=ggplot()+
    ggtitle("Select pattern and degree")+
    theme(legend.position="none")+
    scale_y_continuous("log10(mean squared error)")+
    scale_x_continuous(
      "polynomial degree",
      limits=c(-1, max(degree.vec)+1),
      breaks=degree.vec)+
    scale_color_manual(values=set.colors)+
    make_tallrect(error.dt, "degree", color=NA, fill="blue")+
    geom_line(aes(
      degree, log10(mse.thresh), color=set, group=paste(pattern, set)),
      clickSelects="pattern",
      showSelected="set",
      size=5,
      alpha_off=0.1,
      data=error.dt)+
    geom_point(aes(
      degree, log10(mse.thresh), color=set),
      shape=1,
      fill="white",
      alpha_off=0.1,
      size=4,
      clickSelects="pattern",
      showSelected="set",
      data=best.err)+
    geom_text(aes(
      degree, log10(mse.thresh),
      label=pattern,
      color=set),
      clickSelects="pattern",
      hjust=0,
      showSelected="set",
      data=error.dt[set=="test" & degree==max.degree]),
  funs=ggplot()+
    ggtitle("Selected pattern (points) and degree (blue curve)")+
    xlab("input/feature")+
    ylab("output/label")+
    scale_color_manual(values=set.colors)+
    geom_point(aes(
      x, ynorm, color=set, key=x),
      shape=1,
      fill=NA,
      size=4,
      stroke=1,
      showSelected=c("set","pattern"),
      data=not.grid)+
    geom_line(aes(
      x, pred.thresh, key="pred"),
      data=model.dt[set=="grid"],
      color=model.color,
      showSelected=c("pattern", "degree")),
  duration=list(pattern=1000, degree=1000),
  out.dir="figure-several-interactive",
  title="Overfitting using linear model polynomial degree",
  source="https://github.com/tdhock/cs499-spring2020/blob/master/2020-02-03-capacity/figure-several-interactive.R"))
if(FALSE){
  animint2pages(viz, "2023-12-04-capacity-polynomial-degree-several-patterns")
}

