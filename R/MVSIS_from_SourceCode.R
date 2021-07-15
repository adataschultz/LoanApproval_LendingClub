# MVSIS from Source Code
# Read test data for var screening
F0<-matrix(scan("LoanStatus_screen.txt"), ncol=112, byrow=T)

# Loan status is response
response<-F0[,1]

source("MVSIS.R")

# Define mu for computing the criteria in the simulation
mu<-NULL
for(i in 2:ncol(F0)){
  u<-MV(F0[,i],response)
  mu<-c(mu,u)
  if(i%%100==0) print(i)
}

# Use quantiles with mu for level of rank variables 
q3 <-quantile(mu,0.7)  ### change 0.7 to change the number of selected variables
#q3 <- mu at 0.75
#q3 <- mu at 0.80
mu1 <-cbind(1:(ncol(F0)-1),mu)
name<-read.table("names.csv",sep=",")
name<-name[mu1[,2]>q3]
write(t(name),"names2.csv",sep=",",ncol=length(name))

su<-mu1[mu1[,2]>q3,]
dim(su)

F0<-F0[,-1]
F1<-F0[,su[,1]]
B<-cbind(response,F1)

V<-B
B[15:41,]<-V[19:45,] # 1
B[42:45,]<-V[15:18,] # 0

write(t(B), ncol=ncol(B), file="MVSIS_0.7.txt")

ncol(B)-1  # the first number in C program

################################################################################
###########  Select the data based on set level to run Random Forest ###########
a1<-read.table("MVSIS_0.7.txt")
a1<-as.matrix(a1)
dim(a1)

b<-read.table("names2.csv",sep=",")
b<-as.character.numeric_version(b)
k<-"status"
b<-c(k,b)

write.table(a1,file="MVSIS_0.7_rf.txt",col.names=b)
dim(a1)

################  Random Forest to test predictive performance  ################
library(randomForest)
#Train-Test partition 90:10
k1<-47916
k2<-5324

variable<-read.table("MVSIS_0.7_rf.txt")
variable

# For variables in csv format
write.csv(variable,"MVSIS_0.7.csv" )

# Fit model
rf<-randomForest(factor(status)~.,data=variable[1:k1,],ntree=1000,importance=T,
                 keep.forest=T)
rf

# Use model to predict
m<-rf$predicted
m

n<-predict(rf,variable[(k1+1):(k1+k2),])
n

################################################################################
# Accuracy of train model
k<-0
for (i in 1:k1)
{
  if (variable[i,1]==m[i])
  {
    k=k+1
  }
}
k/k1 # accuracy for fitting dataset

Accuracy.fit <- k/k1 # accuracy for fitting dataset
Accuracy.fit

#Train accuracy: q=0.7 = 0.9779287

################################################################################
# Accuracy of test model
k<-0
for (i in 1:k2)
{
  if (variable[(k1+i),1]==n[i])   ######
  {
    k=k+1
  }
}
k/k2 # accuracy for testing dataset

Accuracy.test <- k/k2 # accuracy for test dataset
Accuracy.test

#Test accuracy: q=0.7 = 0.9789257

################################################################################
### Specificity and Sensitivity
a<-0
for (i in 1:k2)
{
  if (variable[(k1+i),1]==1 & n[i]==1)
  {
    a=a+1
  }
}

b<-0
for (i in 1:k2)
{
  if (variable[(k1+i),1]==0 & n[i]==1)
  {
    b=b+1
  }
}

c<-0
for (i in 1:k2)
{
  if (variable[(k1+i),1]==1 & n[i]==0)
  {
    c=c+1
  }
}

d<-0
for (i in 1:k2)
{
  if (variable[(k1+i),1]==0 & n[i]==0)
  {
    d=d+1
  }
}

sensitivity=a/(a+c)
specificity=d/(b+d)

# Sensitivity : q=0.7 = 0.9972253
# Specificity: q=0.7 = 0.8696133
