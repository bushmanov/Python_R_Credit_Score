setwd('/home/sergey/Py_Credit_Score')
source('./functions/plotImp.R')
library(SIT)

###############################################################################
#
#            Read data into R
#
###############################################################################

# enter feature names
.names <- spl("chk_account_,num_duration_A2,history_,purpose_,num_amount_A5,svn_account_,empl_status_,num_income_share_A8,marital_status_,guarantor_,num_residence_A11,property_,num_age_A13,other_loans_,housing_,num_credits_A16,job_,num_dependants_A18,telephone_,foreign_,status")

# make vector of .colClasses
.num         <- grepl('^num_', .names)  # search for numeric data
.colClasses  <- ifelse(.num, "numeric", "factor")

# read data into R
raw <- read.table(file="./data/german.data",
                  header=F,
                  col.names = .names,
                  colClasses = .colClasses)


###############################################################################
#
#            Explore data
#
###############################################################################

# assign good/bad status
contrasts(raw$status)
levels(raw$status) <- c('good', 'bad')
summary(raw$status)

str(raw)
summary(raw)

hist(raw$num_duration_A2) # credit duration
hist(raw$num_age_A13)     # age of applicant

hist(raw$num_amount_A5)
quantile(raw$num_amount_A5, .8) # .8 quantule for loan amount

library(dplyr)
rawSorted <- filter(raw, num_amount_A5 > quantile(num_amount_A5,.8))
median(rawSorted$num_duration_A2) # median duration for .8 quantile for loan amount
hist(rawSorted$num_duration_A2)

###############################################################################
#
#            Split train / test
#
###############################################################################
# dummies expanded explicitely
.x <- model.matrix(~.-1, data=raw[,-21])
.y <- raw[,21]


library(caTools)
set.seed(1)
ind   <- sample.split(raw$status, SplitRatio = .7)

xTrain <- .x[ ind,]
xTest  <- .x[!ind,]

yTrain <- .y[ ind]
yTest  <- .y[!ind]

# grouped categorical variables for trees methods RF, C5.0 etc.
xTrainG <- raw[ ind,-21]
xTestG  <- raw[!ind,-21]



###############################################################################
#
#            Random Forest
#
###############################################################################
library(doMC)
registerDoMC(2)
library(caret)
models <- list()

ctrl   <- trainControl(method = 'repeatedcv',
                       number=10,
                       repeats=2,
                       allowParallel = T)

gridRF   <- expand.grid(mtry=seq(5,30,by=2))

set.seed(1)
mod_rf <- train(x = xTrain,
                y = yTrain,
                method    ='rf',
                tuneGrid  = gridRF,
                trControl = ctrl)

models$rf <- mod_rf
plot(mod_rf)

# grouped RF
set.seed(1)
mod_rfG <- train(x = xTrainG,
                 y = yTrain,
                 method    ='rf',
                 tuneGrid  = gridRF,
                 trControl = ctrl)

models$rfG <- mod_rfG

###############################################################################
#
#            Conditional inference trees
#
###############################################################################
#names(getModelInfo())
#modelLookup('cforest')

set.seed(1)
mod_cf <- train(x = xTrain,
                y = yTrain,
                method    ='cforest',
                tuneGrid  = gridRF,
                trControl = ctrl)

models$cf <- mod_cf

# grouped cforest
set.seed(1)
mod_cfG <- train(x = xTrainG,
                 y = yTrain,
                 method    ='cforest',
                 tuneGrid  = gridRF,
                 trControl = ctrl)

models$cfG <- mod_cfG

###############################################################################
#
#            Boosted Trees
#
###############################################################################

#modelLookup('gbm')

gbmGrid <- expand.grid(interaction.depth = c(4,6,8),
                       n.trees = 1000,
                       shrinkage = c(0.01,0.1),
                       n.minobsinnode=c(3,5,10))

set.seed(1)
mod_gbm <- train(x = xTrain,
                 y = yTrain,
                 method='gbm',
                 tuneGrid = gbmGrid,
                 verbose=F,
                 trControl = ctrl)

models$gbm <- mod_gbm
plot(mod_gbm)

# grouped gmv
set.seed(1)
mod_gbmG <- train(x = xTrainG,
                  y = yTrain,
                  method='gbm',
                  tuneGrid = gbmGrid,
                  verbose=F,
                  trControl = ctrl)

models$gbmG <- mod_gbmG

###############################################################################
#
#            С5.0
#
###############################################################################

# names(getModelInfo())
# modelLookup("C5.0")

set.seed(1)
c5_mod <- train(x = xTrain,
                y = yTrain,
                method     = "C5.0",
                tuneLength = 10,
                trControl  = ctrl)

models$c5 <- c5_mod
plot(c5_mod)

# grouped C5.0
set.seed(1)
c5_modG <- train(x = xTrainG,
                 y = yTrain,
                 method     = "C5.0",
                 tuneLength = 10,
                 trControl  = ctrl)

models$c5G <- c5_modG

###############################################################################
#
#            Regularized GLM (glmnet)
#
###############################################################################

# names(getModelInfo())
# modelLookup("glmnet")

grid <- expand.grid(alpha = seq(0,.2,by=.01),
                    lambda = 10^(seq(-3,0,by=.5)))

set.seed(1)
glmnet_mod <- train(x = xTrain,
                    y = yTrain,
                    method = "glmnet",
                    tuneGrid = grid,
                    preProcess = c('center', 'scale'),
                    trControl = ctrl)

models$glm <- glmnet_mod
plot(glmnet_mod)

###############################################################################
#
#            Model comparison
#
###############################################################################

rf_imp   <- varImp(models$rf )
rf_impG  <- varImp(models$rfG) # '*G$' stands for 'grouped' dummies
gbm_imp  <- varImp(models$gbm)
gbm_impG <- varImp(models$gbmG)
c5_imp   <- varImp(models$c5 , metric='splits')
c5_impG  <- varImp(models$c5G, metric='splits')
coefGLM <- coef(glmnet_mod$finalModel, glmnet_mod$bestTune$lambda) # coefs for best glm
glm_imp <- list()
glm_imp$importance <- abs(coefGLM[order(abs(coefGLM), decreasing = T),][1:10])

par(mfrow=c(3,2))
plotImp(rf_imp,  main = '  RF  var importance')
plotImp(rf_impG, main = '  RFG var importance')
plotImp(gbm_imp, main = '  GBM var importance')
plotImp(gbm_impG,main = ' GBMG var importance')
plotImp(c5_imp,  main = ' C5.0 var importance')
plotImp(c5_impG, main = 'C5.0G var importance')
#plotImp(glm_imp, main = ' GLM var importance')
dev.off()


resamp <- resamples(models)
summary(resamp)
bwplot(resamp, metric='Accuracy')

xyplot(resamp)
summary(diff(resamp))


###############################################################################
#
#            Accuracy on held-out sample
#
###############################################################################

accuracy <- vector(mode='numeric')

for (i in names(models)) {
    if (grepl('G$', i)) {
        accuracy[i] <- mean(predict(models[[i]], xTestG) == yTest)
    } else {
        accuracy[i] <- mean(predict(models[[i]], xTest ) == yTest)
        }
}

dotchart(sort(accuracy), main="Model Accuracy")


#
# ###############################################################################
# #
# #            Incorporating cost matrix
# #
# ###############################################################################
#
# # names(getModelInfo())
# # modelLookup("C5.0")
#
# cost <- matrix(c(0,5,1,0), byrow = T, nrow=2)
# colnames(cost) <- c('good', 'bad') # ground truth
# rownames(cost) <- c('good', 'bad') # predictions
#
# set.seed(1)
# c5cost_mod <- train(x=xTrain,
#                     y=yTrain,
#                     method     = "C5.0",
#                     tuneLength = 10,
#                     cost       = cost,
#                     trControl  = ctrl)
#
# # c5cost_mod$bestTune
# # c5cost_imp <- varImp(c5cost_mod, metric = 'splits')
# # plotImp(c5cost_imp$importance)
#
#
