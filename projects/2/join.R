evals.dt <- data.table::fread(
  "cs499 spring 2020 project 2 grades - group-evals.csv")
fill <- function(x){
  x.fac <- factor(x)
  levs <- levels(x.fac)
  x.int <- as.integer(x.fac)
  x.int[x==""] <- NA
  x.fill <- data.table::nafill(x.int, type="locf")
  factor(x.fill, seq_along(levs), levs)
}
evals.dt[, eval.fill := fill(evaluator)]
member.percents <- evals.dt[, .(
  mean.percent=mean(member.percent)
), by=.(member.name)]

scores.dt <- data.table::fread(
  "cs499 spring 2020 project 2 grades - scores.csv"
)[group != ""]
member.totals <- scores.dt[, nc::capture_all_str(
  group,
  member.name="[^-]+"),
  by=.(group, TOTAL)]

join.dt <- member.percents[member.totals, on="member.name"]
join.dt[is.na(mean.percent), mean.percent := 100]
join.dt[, TOTAL.scaled := TOTAL*mean.percent/100]
join.dt[, max := ifelse(TOTAL>120, TOTAL, 120)]
join.dt[, grade := ifelse(TOTAL.scaled>max, max, TOTAL.scaled)]
join.dt[order(member.name), .(member.name, grade)]

scores.tall <- nc::capture_melt_single(
  scores.dt,
  names=".*-.*")
scores.tall[, score := as.integer(value)]
members <- scores.tall[, data.table::data.table(score, nc::capture_all_str(
  names,
  member.name="[^-]+")),
  by=names]
clean <- function(x)gsub("\\s", "", x)
member.evals <- members[evals.dt, on="member.name"]
mean.evals <- member.evals[, .(
  percent=mean(member.score),
  evals=.N
), by=member.name]
join.dt <- mean.evals[members, on="member.name"]
max.score <- 250
join.dt[, grade := score*percent/100]
join.dt[, gcap := ifelse(grade>max.score, max.score, grade)]
join.dt[order(member.name), .(member.name, gcap, grade, score, percent)]
