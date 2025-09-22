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
lm.poly.deg <- "polynomial degree"
num.nn <- "number of neighbors"
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
  data.table(
    model="Linear (polynomial basis)", first=1,
    regularization=lm.poly.deg, color="blue", size=3, hjust=0, fun="max"),
  data.table(
    model="Nearest neighbors", first=3,
    regularization=num.nn, color="green", size=2, hjust=1, fun="min"))
model.colors <- reg.dt[, structure(color, names=model)]
model.sizes <- reg.dt[, structure(size, names=model)]
expand <- 0.1
not.grid <- model.dt[set!="grid"]
model.dt[, pred.thresh := ifelse(
  pred.y < min(not.grid$ynorm)-expand, -Inf,
  ifelse(pred.y > max(not.grid$ynorm)+expand, Inf, pred.y))]
tallrect.dt <- unique(error.dt[, .(regularization, parameter)])
duration.list <- list(pattern=1000)
height.pixels <- 500
gg.list <- list(
  Data=ggplot()+
    ggtitle("Selected pattern (points) and models (curves)")+
    xlab("input/feature")+
    ylab("output/label")+
    theme_animint(height=height.pixels, rowspan=2, last_in_row=TRUE)+
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
      size=model,
      key=model,
      group=model,
      color=model),
      data=model.dt[set=="grid"][reg.dt, on="regularization"],
      showSelected=c("pattern", regularization="parameter"))
)
selectize.list <- list()
first.list <- list(pattern="linear")
for(model.i in 1:nrow(reg.dt)){
  reg.row <- reg.dt[model.i]
  selectize.list[[reg.row$regularization]] <- TRUE
  first.list[[reg.row$regularization]] <- reg.row$first
  gg_name <- sub(" .*", "", reg.row$model)
  duration.list[[reg.row$regularization]] <- 1000
  model.err <- error.dt[reg.row, on="regularization"][
  , (reg.row$regularization) := parameter
  ][]
  model.best <- best.err[reg.row, on="regularization"]
  FUN <- get(reg.row$fun)
  text_x <- FUN(model.err$parameter)
  model.text <- model.err[parameter == text_x & set=="test"]
  extra_x <- text_x-2*(reg.row$hjust-0.5)#how far to expand one side for text labels.
  limits_x <- range(c(extra_x, model.err$parameter))
  gg.list[[gg_name]] <- ggplot()+
    ggtitle(paste("Select", reg.row$regularization))+
    theme(legend.position="none")+
    theme_animint(height=height.pixels/2)+
    scale_y_continuous("log10(mean square error)")+
    scale_x_continuous(
      reg.row$regularization,
      limits=limits_x+0.5*c(-1,1),#how far to expand beyond data on both sides.
      breaks=unique(model.err$parameter))+
    scale_color_manual(values=set.colors)+
    make_tallrect(
      model.err,
      reg.row$regularization,
      color=NA,
      fill=model.colors[[reg.row$model]])+
    geom_line(aes(
      parameter, log10(mse.thresh), color=set, group=paste(pattern, set)),
      clickSelects="pattern",
      size=5,
      alpha_off=0.1,
      showSelected=c("model", "set"),
      data=model.err)+
    geom_point(aes(
      parameter, log10(mse.thresh), color=set),
      shape=1,
      fill="white",
      alpha_off=0.1,
      size=4,
      clickSelects="pattern",
      showSelected=c("model", "set"),
      data=model.best)+
    geom_label_aligned(aes(
      parameter, log10(mse.thresh),
      hjust=hjust,
      label=pattern,
      color=set),
      clickSelects="pattern",
      showSelected=c("model", "set"),
      data=model.text)
}
viz <- animint(
  duration=duration.list,
  out.dir="figure-polynomial-neighbors-table-interactive",
  title="Overfitting using linear model polynomial degree and nearest neighbors, HTML table layout",
  first=first.list,
  selectize=selectize.list,
  source="https://github.com/tdhock/cs499-spring2020/blob/master/2020-02-03-capacity/figure-polynomial-neighbors-table-interactive.R")
plot.ord <- c("Linear", "Data", "Nearest")
for(plot.i in seq_along(plot.ord)){
  gg_name <- plot.ord[[plot.i]]
  viz[[gg_name]] <- gg.list[[gg_name]]
}
viz
if(FALSE){
  animint2pages(viz, "2025-09-22-degree-neighbors-table", chromote_sleep_seconds = 5)
}

