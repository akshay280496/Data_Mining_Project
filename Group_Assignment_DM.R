library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(lattice)
library(DataExplorer)
library(factoextra)
library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)
library(Metrics)
library(ROCit)
library(kableExtra)

setwd("D:/BABI/BABI-5th Residency/Data Mining/Group Assignment")
getwd()
library(readxl) #package to read the excel file
bank <- read_excel("Thera Bank-Data Set.xlsx", sheet = "Bank_Personal_Loan_Modelling")
View(bank)
dim(bank) #checking the dimensions

any(is.na(bank)) #checking for missing values

sapply(bank, function(x){sum(is.na(x))}) #checking columns which have missing values

bank[is.na(bank)]=0  #check again after replacing with 0
any(is.na(bank)) #re-checking for missing values

dim(bank)
str(bank) #structure of the bank dataset

summary(bank)

bank = bank[,-c(1,5)] #dropping the first and 5th columns of the dataset
View(bank)

#converting multiple columns into factor columns

col = c("Education","Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard")
bank[col] = lapply(bank[col], factor)

#converting education into ordinals

bank$Education = factor(bank$Education, levels = c("1","2","3"), order = TRUE)

#Abbreviating variable names

bank = bank%>% rename(Age = "Age (in years)", Experience = "Experience (in years)", 
                      Income = "Income (in K/month)")
View(bank)

head(bank[bank$Experience<0,]) #checking for negative values in experience col

bank$Experience = abs(bank$Experience) #modulo function for the negative values

dim(bank)

summary(bank)

plot_intro(bank)#EDA

plot_histogram(bank)

plot_density(bank,geom_density_args = list(fill="blue", alpha=0.5))

plot_boxplot(bank, by = "Education", geom_boxplot_args = list("outlier.color" = "green"))

plot_boxplot(bank, by = "Personal Loan", geom_boxplot_args = list("outlier.color" = "blue"))

p1 = ggplot(bank, aes(Income, fill= "Personal Loan")) + geom_density(alpha=0.5)
p2 = ggplot(bank, aes(Mortgage, fill= "Personal Loan")) + geom_density(alpha=0.5)
p3 = ggplot(bank, aes(Age, fill= "Personal Loan")) + geom_density(alpha=0.5)

p4 = ggplot(bank, aes(Experience, fill= "Personal Loan")) + geom_density(alpha=0.5)
p5 = ggplot(bank, aes(Income, fill= Education)) + geom_histogram(alpha=0.5, bins = 70)
p6 = ggplot(bank, aes(Income, Mortgage, color = "Personal Loan")) + 
  geom_point(alpha = 0.5)
grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)

#number of personal loans taken w.r.t. education


ggplot(bank, aes(Education,fill= bank$`Personal Loan`)) + 
  geom_bar(stat = "count", position = "dodge") +
  geom_label(stat = "count", aes(label= ..count..), 
             size = 3, position = position_dodge(width = 0.9), vjust=-0.15)+
  scale_fill_manual("Personal Loan", values = c("0" = "blue", "1" = "green"))+
  theme_minimal()

summary(bank$`Personal Loan`)

#number of personal loans taken w.r.t. credit card

ggplot(bank, aes(Income,y = CCAvg, color = bank$`Personal Loan`)) + 
  geom_point(size = 1)

#number of personal loans taken w.r.t. mortgage

ggplot(bank, aes(Income,y = Mortgage, color = bank$`Personal Loan`)) + 
  geom_point(size = 1)

#Clustering begins

bank.clust = bank %>% select_if(is.numeric)

bank.scale = scale(bank.clust, center = TRUE)  #scaling the cluster

bank.dist = dist(bank.scale, method = "euclidean") #calculating the euclidean distance

## checking optimal number of clusters to categorize dataset 

p12 = fviz_nbclust(bank.scale, kmeans, method = "silhouette", k.max = 5) # k-means clustering is used
p21 = fviz_nbclust(bank.scale, kmeans, method = "wss", k.max = 5)

grid.arrange(p12, p21, ncol=2)

#k-means clustering

set.seed(8787)
bank.clusters = kmeans(bank.scale, 3, nstart = 10)

fviz_cluster(bank.clusters, bank.scale, geom = "point", 
             ellipse = TRUE, pointsize = 0.2, ) + theme_minimal()

#splitting dataset into train and test

set.seed(1233)

## sampling 70% of data for training the algorithms using random sampling 

bank.index = sample(1:nrow(bank), nrow(bank)*0.70)
bank.train = bank[bank.index,]
bank.test = bank[-bank.index,]

dim(bank.test)

dim(bank.train)

table(bank.train$`Personal Loan`)

table(bank.test$`Personal Loan`)

#CART model

set.seed(233)

cart.model.gini = rpart(bank.train$`Personal Loan`~., data = bank.train, method = "class",
                        parms = list(split="gini")) 

## checking the complexity parameter 
plotcp(cart.model.gini)   

## plotting the classification tree 
rpart.plot(cart.model.gini, cex =0.6)

## checking the cptable to gauge the best crossvalidated error and correspoding
## Complexity paramter 
cart.model.gini$cptable

## checking for the variable importance for splitting of tree
cart.model.gini$variable.importance

#Pruned CART Tree

## prunning the tree using the best complexity parameter
pruned.model = prune(cart.model.gini, cp = 0.015)

## plotting the prunned tree
rpart.plot(pruned.model, cex=0.65)

#CART prediction

cart.pred = predict(pruned.model, bank.test, type = "prob")

cart.pred.prob.1 = cart.pred[,1]
head(cart.pred.prob.1, 10)

## setting the threshold for probabilities to be considered as 1 
threshold = 0.70

bank.test$loanprediction = ifelse(cart.pred.prob.1 >= threshold, 1, 0)

bank.test$loanprediction = as.factor(bank.test$loanprediction)

Cart.Confusion.Matrix = confusionMatrix(bank.test$loanprediction, 
                                        reference = bank.test$`Personal Loan`, positive = "1")
Cart.Confusion.Matrix

#Random Forest

set.seed(1233)

RF = randomForest(formula = bank.test$`Personal Loan`~(bank.test$Age+bank.test$Experience+
                                                          bank.test$Income+bank.test$`Family members`+
                                                          bank.test$CCAvg+bank.test$Education+
                                                          bank.test$Mortgage+
                                                          bank.test$`Securities Account`+
                                                          bank.test$`CD Account`+
                                                          bank.test$Online+bank.test$CreditCard), 
                  data = bank.test)
print(RF)

## Print the error rate

err = RF$err.rate
head(err)

## out of bag error 
oob_err = err[nrow(err), "OOB"]
print(oob_err)  ## depicts the final out of bag error for all the samples

## plot the OOB error 

plot(RF)
legend(x="topright", legend = colnames(err), fill = 1:ncol(err))

#Prediction for Random Forest package

ranfost.pred = predict(RF, bank.test, type = "prob")[,1]

bank.test$RFpred = ifelse(ranfost.pred>=0.7,"1","0")

bank.test$RFpred = as.factor(bank.test$RFpred)

levels(bank.test$RFpred)

RFConf.Matx = confusionMatrix(bank.test$RFpred, bank.test$`Personal Loan`, positive = "1")
RFConf.Matx

table(bank.test$`Personal Loan`)

#tuning the Random Forest algorithm

set.seed(333)

tunedRF = tuneRF(x = bank.test[,-8],
                        y= bank.test$`Personal Loan`,
                        ntreeTry = 501, doBest = T)

print(tunedRF)

# #plotting of ROC curve
# 
# Prediction.Labels = as.numeric(range.pred$predictions)
# Actual.Labels = as.numeric(bank.test$Personal.Loan)
# 
# roc_Rf = rocit(score = Prediction.Labels, class = Actual.Labels)
# 
# plot(roc_Rf)




