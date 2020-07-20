library("tree")
library("adabag")
library("randomForest")

spam <- read.csv("spambase.data",header = F)
str(spam)
names(spam) <- c("word_freq_make","word_freq_address","word_freq_all","word_freq_3d",
                         "word_freq_our","word_freq_over",
                         "word_freq_remove", "word_freq_internet",
                         "word_freq_order", "word_freq_mail", "word_freq_receive",
                         "word_freq_will",
                         "word_freq_people", "word_freq_report",
                         "word_freq_addresses", "word_freq_free",
                         "word_freq_business",
                         "word_freq_email",
                         "word_freq_you",
                         "word_freq_credit",
                         "word_freq_your",
                         "word_freq_font",       "word_freq_000",
                         "word_freq_money",
                         "word_freq_hp",
                         "word_freq_hpl",
                         "word_freq_george",
                         "word_freq_650",
                         "word_freq_lab",
                         "word_freq_labs",
                         "word_freq_telnet",
                         "word_freq_857",
                         "word_freq_data",
                         "word_freq_415",
                         "word_freq_85",
                         "word_freq_technology",
                         "word_freq_1999",
                         "word_freq_parts",
                         "word_freq_pm",
                         "word_freq_direct",
                         "word_freq_cs",
                         "word_freq_meeting",
                         "word_freq_original",
                         "word_freq_project",
                         "word_freq_re",
                         "word_freq_edu",
                         "word_freq_table",
                         "word_freq_conference",    "char_freq_;",
                         "char_freq_(",
                         "char_freq_[",
                         "char_freq_!", "char_freq_$",
                         "char_freq_#",  "capital_run_length_average",
                         "capital_run_length_longest",
                         "capital_run_length_total","spam")

spam$spam <- as.factor(spam$spam)
spam <- data.frame(spam)

prop.table(table(spam$spam))
#email(0)      spam(1) 
#0.6059552 0.3940448 

#Splitting the data into train and test
set.seed(1234)
indx <- sample(1:nrow(spam),2301,replace = FALSE)
TrainData_spam <- spam[indx,]
TestData_spam <- spam[-indx,]

intersect(TrainData_spam,TestData_spam)
prop.table(table(TrainData_spam$spam))
prop.table(table(TestData_spam$spam))

# Decision Tree

spam.tree <- tree(spam ~ ., data=TrainData_spam)
plot(spam.tree)
text(spam.tree,cex = 0.6)
#Cross - Validation
spam.tree.cv <- cv.tree(spam.tree , FUN = prune.misclass)
plot(spam.tree.cv)
spam.tree.cv

size <- spam.tree.cv$size[which.min(spam.tree.cv$dev)]
size

#Pruning

spam.tree.prune <- prune.tree(spam.tree,best = size)
plot(spam.tree.prune)
text(spam.tree.prune,cex=0.5)
summary(spam.tree.prune)

predict.tree <- predict(spam.tree.prune,newdata=TestData_spam,type="class")
mean(predict.tree != TestData_spam$spam)

#Bagging

spam.bag <- bagging(spam~., data=TrainData_spam,mfinal = 100,importance= TRUE)
imp <- sort(spam.bag$importance,decreasing = TRUE)
par(mar=c(12,2,1,1)+.1)
barplot(imp[imp>0],las=2,ylim=c(0,40))

predict.bag <- predict.bagging(spam.bag,newdata = TestData_spam)
predict.bag$confusion
predict.bag$error

#Random Forest 

spam.rf <- randomForest(spam ~ ., data=TrainData_spam,mtry = sqrt(ncol(spam)-1), ntree=100,importance=TRUE, proximity=TRUE)
spam.rf
importance(spam.rf)
varImpPlot(spam.rf,main="Importance of the variables")
predict.rf <- predict(spam.rf,newdata = TestData_spam)
library(caret)
library(e1071)
confusionMatrix(predict.rf, TestData_spam$spam)


