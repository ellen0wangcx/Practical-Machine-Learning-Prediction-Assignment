require(data.table)
#setInternet2(TRUE)
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train <- fread(url)

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
test <- fread(url)

isAnyMissing <- sapply(test, function (x) any(is.na(x) | x == ""))
isPredictor <- !isAnyMissing & grepl("belt|[^(fore)]arm|dumbbell|forearm", names(isAnyMissing))
predCandidates <- names(isAnyMissing)[isPredictor]
predCandidates

varToInclude <- c("classe", predCandidates)
train <- train[, varToInclude, with=FALSE]
dim(train)
names(train)

train <- train[, classe := factor(train[, classe])]
train[, .N, classe]

require(caret)
seed <- as.numeric(as.Date("2014-10-26"))
set.seed(seed)
inTrain <- createDataPartition(train$classe, p=0.6)
DTrain <- train[inTrain[[1]]]
DProbe <- train[-inTrain[[1]]]

X <- DTrain[, predCandidates, with=FALSE]
preProc <- preProcess(X)
preProc
XCS <- predict(preProc, X)
DTrainCS <- data.table(data.frame(classe = DTrain[, classe], XCS))

X <- DProbe[, predCandidates, with=FALSE]
XCS <- predict(preProc, X)
DProbeCS <- data.table(data.frame(classe = DProbe[, classe], XCS))

nzv <- nearZeroVar(DTrainCS, saveMetrics=TRUE)
if (any(nzv$nzv)) nzv else message("No variables with near zero variance")


histGroup <- function (data, regex) {
  col <- grep(regex, names(data))
  col <- c(col, which(names(data) == "classe"))
  require(reshape2)
  n <- nrow(data)
  DMelted <- melt(data[, col, with=FALSE][, rownum := seq(1, n)], id.vars=c("rownum", "classe"))
  require(ggplot2)
  ggplot(DMelted, aes(x=classe, y=value)) +
    geom_violin(aes(color=classe, fill=classe), alpha=1/2) +
    #     geom_jitter(aes(color=classe, fill=classe), alpha=1/10) +
    #     geom_smooth(aes(group=1), method="gam", color="black", alpha=1/2, size=2) +
    facet_wrap(~ variable, scale="free_y") +
    scale_color_brewer(palette="Spectral") +
    scale_fill_brewer(palette="Spectral") +
    labs(x="", y="") +
    theme(legend.position="none")
}
histGroup(DTrainCS, "belt")
histGroup(DTrainCS, "[^(fore)]arm")
histGroup(DTrainCS, "dumbbell")
histGroup(DTrainCS, "forearm")


require(parallel)
require(doParallel)
library(doParallel)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)


ctrl <- trainControl(classProbs=TRUE,
                     savePredictions=TRUE,
                     allowParallel=TRUE)

method <- "rf"
system.time(trainingModel <- train(classe ~ ., data=DTrainCS, method=method))

stopCluster(cl)

trainingModel
hat <- predict(trainingModel, DTrainCS)
confusionMatrix(hat, DTrain[, classe])
hat <- predict(trainingModel, DProbeCS)
confusionMatrix(hat, DProbeCS[, classe])

varImp(trainingModel)
trainingModel$finalModel

save(trainingModel, file="trainingModel.RData")

load(file="trainingModel.RData", verbose=TRUE)

DTestCS <- predict(preProc, test[, predCandidates, with=FALSE])
hat <- predict(trainingModel, DTestCS)
DTest <- cbind(hat , test)
subset(DTest, select=names(DTest)[grep("belt|[^(fore)]arm|dumbbell|forearm", names(DTest), invert=TRUE)
