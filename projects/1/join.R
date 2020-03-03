evals.dt <- data.table::fread(
  "cs499 spring 2020 project 1 grades - group-evals.csv")
evals.dt[["member.name"]] <- clean(evals.dt[["member.name\n"]])
evals.dt[["member.score"]] <- evals.dt[["member.score\n"]]
evals.dt[, evaluator := clean(evaluator)]
scores.dt <- data.table::fread(
  "cs499 spring 2020 project 1 grades - scores.csv")[20]
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
