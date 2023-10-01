# MVSIS from Source Code
# Read test data for var screening
F0<-matrix(scan("LoanStatus_screen.txt"), ncol=112, byrow=T)

# Loan status is response
response<-F0[,1]

source("MVSIS.R")

# Define mu for computing the criteria in the simulation
mu <- NULL
for(i in 2:ncol(F0)){
    u <- MV(F0[,i], response)
    mu <- c(mu,u)
    if(i%%100==0) print(i)
}

# Use quantiles with mu for level of rank variables 
q3 <- quantile(mu, 0.7)  ### change 0.7 to change the number of selected variables
mu <- cbind(1:(ncol(F0)-1), mu)
name <- read.table('names.csv', sep=',')
name <- name[mu[,2] > q3]
write(t(name), 'names2.csv', sep=',', ncol=length(name))

su <- mu[mu[,2] > q3,]
dim(su)

F0 <- F0[,-1]
F1 <- F0[,su[,1]]
B <- cbind(response,F1)

V <- B
B[15:41,] <- V[19:45,] # 1
B[42:45,] <- V[15:18,] # 0

write(t(B), ncol=ncol(B), file='MVSIS_0.7.txt')

ncol(B) - 1 # the first number in C program

################################################################################
###########  Select the data based on set level to run Random Forest ###########
################################################################################
a1 <- read.table('MVSIS_0.7.txt')
a1 <- as.matrix(a1)
dim(a1)

b <- read.table('names2.csv', sep=',')
b <- as.character.numeric_version(b)
k <- 'status'
b <- c(k,b)

write.table(a1, file='MVSIS_0.7_rf.txt', col.names=b)
dim(a1)

################################################################################
################  Random Forest to test predictive performance  ################
################################################################################
library(randomForest)
library(doParallel)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)
getDoParWorkers()

# Read data
variable <- read.table('MVSIS_0.7_rf.txt')
variable = variable[sample(1:nrow(variable)),]
colnames(variable)[colnames(variable) == 'status'] = 'loan_status'
str(variable)

# For variables in csv format
write.csv(variable,"MVSIS_0.7.csv")

# Define train/test sets
k1 <- 194612
k2 <- 21624

# Fit model
rf <- randomForest(factor(loan_status) ~ .,
                   data=variable[1:k1,],
                   ntree=100,
                   importance=T,
                   keep.forest=T)
rf
stopCluster(cl)

################################################################################
# Use model to predict
m <- rf$predicted
n <- predict(rf, variable[(k1+1):(k1+k2),])

# Accuracy of train model
k <- 0
for (i in 1:k1){
    if (variable[i,1]==m[i]){
        k=k+1
    }
}

Accuracy.fit <- k / k1

cat(sprintf('Train accuracy = %s\n', Accuracy.fit))

k <- 0
for (i in 1:k2){
    if (variable[(k1+i),1]==n[i]){
        k=k+1
    }
}

Accuracy.test <- k / k2
cat(sprintf('Test accuracy = %s\n', Accuracy.test))

################################################################################
# Specificity and Sensitivity
a <- 0
for (i in 1:k2){
    if (variable[(k1+i),1]==1 & n[i]==1){
        a = a+1
    }
}

b <- 0
for (i in 1:k2){
    if (variable[(k1+i),1]==0 & n[i]==1){
        b = b+1
    }
}

c <- 0
for (i in 1:k2){
    if (variable[(k1+i),1]==1 & n[i]==0){
        c = c+1
    }
}

d <- 0
for (i in 1:k2){
    if (variable[(k1+i),1]==0 & n[i]==0){
        d = d+1
    }
}

sensitivity = d / (b + d)
cat(sprintf('Sensitivity = %s\n', sensitivity))

specificity = a / (a + c)
cat(sprintf('Specificity = %s\n', specificity))

################################################################################
train <- k1
test <- k2

# Train
train$loan_status <- as.factor(train$loan_status)

pred = predict(rf, newdata=train[-1])

pred1 <- as.factor(pred)

confusionMatrix(data=pred1, reference=train$loan_status)

# Test
test$loan_status <- as.factor(test$loan_status)

pred = predict(rf, newdata=test[-1])

pred1 <- as.factor(pred)

confusionMatrix(data=pred1, reference=test$loan_status)
################################################################################