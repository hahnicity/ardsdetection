library("data.table")
library("e1071")
ardsfiles = Sys.glob("/Users/greg/Documents/bedside-to-cloud/ardsproject/ardscohort/*/*_breath_meta*.csv")
copdfiles = Sys.glob("/Users/greg/Documents/bedside-to-cloud/ardsproject/copdcohort/*/*_breath_meta*.csv")
controlfiles = Sys.glob("/Users/greg/Documents/bedside-to-cloud/ardsproject/controlcohort/*/*_breath_meta*.csv")
ardscopdfiles = Sys.glob("/Users/greg/workspace/pythonDocuments/bedside-to-cloud/ardsproject/ardscopdcohort/*/*_breath_meta*.csv")

read_and_filter <- function(filename) {
  patient = fread(filename)
  patient$V20 <- NULL
  patient = patient[complete.cases(patient),]
  patient$min_vent = patient$inst_RR * patient$TVi
  return (patient)
}

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y <- y[complete.cases(y)]
  return(y)
}

ards = data.frame()
copd = data.frame()
control = data.frame()
ardscopd = data.frame()

for(f in ardsfiles) {ards = rbind(ards, read_and_filter(f))}
for(f in copdfiles) {copd = rbind(copd, read_and_filter(f))}
for(f in controlfiles) {control = rbind(control, read_and_filter(f))}
for(f in ardscopdfiles) {ardscopd = rbind(ardscopd, read_and_filter(f))}

ards$min_vent = ards$TVi * ards$inst_RR
copd$min_vent = copd$TVi * copd$inst_RR
control$min_vent = control$TVi * control$inst_RR
ardscopd$min_vent = ardscopd$TVi * ardscopd$inst_RR

ards$V20 <- NULL
copd$V20 <- NULL
ardscopd$V20 <- NULL
control$V20 <- NULL

ards = ards[-which(ards$min_vent < 0)]
copd = copd[-which(copd$min_vent < 0)]
control = control[-which(control$min_vent < 0)]
ardscopd = ardscopd[-which(ardscopd$min_vent < 0)]

hist(subset(ards$min_vent, ards$min_vent < 20000), breaks=100)
hist(subset(copd$min_vent, copd$min_vent < 20000), breaks=100)
hist(subset(ardscopd$min_vent, ardscopd$min_vent < 20000), breaks=100)
hist(subset(control$min_vent, control$min_vent < 20000), breaks=100)

hist(subset(log(ards$min_vent), ards$min_vent < 20000), breaks=100)
hist(subset(log(copd$min_vent), copd$min_vent < 20000), breaks=100)
hist(subset(log(ardscopd$min_vent), ardscopd$min_vent < 20000), breaks=100)
hist(subset(log(control$min_vent), control$min_vent < 20000), breaks=100)

ards = ards[-which(ards$dyn_compliance < 0)]
copd = copd[-which(copd$dyn_compliance < 0)]
ardscopd = ardscopd[-which(ardscopd$dyn_compliance < 0)]
control = control[-which(control$dyn_compliance < 0)]

hist(subset(control$minF_to_zero, control$minF_to_zero < 200), breaks=100)
hist(subset(ards$minF_to_zero, ards$minF_to_zero < 200), breaks=100)
hist(subset(copd$minF_to_zero, copd$minF_to_zero < 200), breaks=100)
hist(subset(log(ardscopd$minF_to_zero), ardscopd$minF_to_zero < 200), breaks=100)

hist(ards$inst_RR, breaks=100)
hist(copd$inst_RR, breaks=100)
hist(control$inst_RR, breaks=100)

hist(subset(ards$`pef_+0.16_to_zero`, ards$`pef_+0.16_to_zero` < 100), breaks=100)
hist(subset(copd$`pef_+0.16_to_zero`, copd$`pef_+0.16_to_zero` < 100), breaks=100)
hist(subset(control$`pef_+0.16_to_zero`, control$`pef_+0.16_to_zero` < 100), breaks=100)
hist(subset(ardscopd$`pef_+0.16_to_zero`, ardscopd$`pef_+0.16_to_zero` < 100), breaks=100)

hist(remove_outliers(ards$dyn_compliance), breaks=100)
hist(remove_outliers(copd$dyn_compliance), breaks=100)
hist(remove_outliers(ardscopd$dyn_compliance), breaks=100)
hist(remove_outliers(control$dyn_compliance), breaks=100)

# per patient level

pt1 = read_and_filter(controlfiles[1])
pt84 = read_and_filter(copdfiles[1])
pt8 = read_and_filter(ardsfiles[1])

analyze_data <- function(data) {
  plot(data)
  print(mean(data))
  print(median(data))
  print(var(data))
  print(kurtosis(data))
  print(skewness(data))
}

examine_pef_to_zero <- function(files) {
  for (f in files) {
    patient <- read_and_filter(f)
    print(f)
    no_outliers <- remove_outliers(patient$minF_to_zero)
    analyze_data(no_outliers)
    Sys.sleep(5)
  }
}

examine_pef_plus_to_zero <- function(files) {
  for (f in files) {
    patient <- read_and_filter(f)
    print(f)
    no_outliers <- remove_outliers(patient$`pef_+0.16_to_zero`)
    analyze_data(no_outliers)
    Sys.sleep(5)
  }
}

examine_min_vent <- function(files) {
  for (f in files) {
    patient <- read_and_filter(f)
    print(f)
    no_outliers <- remove_outliers(patient$min_vent)
    analyze_data(no_outliers)
    Sys.sleep(5)
  }
}

examine_dyn_compliance <- function(files) {
  for (f in files) {
    patient <- read_and_filter(f)
    print(f)
    no_outliers <- remove_outliers(patient$dyn_compliance)
    analyze_data(no_outliers)
    Sys.sleep(5)
  }
}

examine_brunner <- function(files) {
  for (f in files) {
    patient <- read_and_filter(f)
    print(f)
    no_outliers <- remove_outliers(patient$brunner)
    analyze_data(no_outliers)
    Sys.sleep(5)
  }
}

examine_vol_at_time <- function(files) {
  for (f in files) {
    patient <- read_and_filter(f)
    print(f)
    no_outliers <- remove_outliers(patient$vol_at_.5_sec)
    analyze_data(no_outliers)
    Sys.sleep(5)
  }
}

remove_cols <- function(df) {
  df$BN <- NULL
  df$ventBN <- NULL
  df$BS <- NULL
  df$BS <- NULL
  df$IEnd <- NULL
  df$x01 <- NULL
  df$x02 <- NULL
  df$TVi2 <- NULL
  df$TVe2 <- NULL
  df$TVi1 <- NULL
  df$TVe1 <- NULL
  df$vent_mode <- NULL
  return (df)
}

boxplot(remove_outliers(pt8ards$minF_to_zero))

