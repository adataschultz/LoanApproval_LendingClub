################################################################################
############################## EDA and Variable Selection ######################
################################################################################
# Install required packages for Loan-Status
library(DataExplorer)
library(ggplot2)
library(tidyr)
library(plyr)
library(dplyr)
library(fastDummies)
library(caret)
library(VariableScreening)
library(doParallel)
library(Boruta)

# Set working directory
#setwd("path")

# Read data
data <- read.csv("loan_Master.csv")
dim(data) 
str(data)

# Missing data
final_data <- data
test_data <- final_data
j <- 1
for (i in final_data){
  if ((is.numeric(final_data[,j]) == FALSE))
    test_data[,j] <- as.data.frame(trimws(final_data[,j]))
  j <- j + 1
}

# Replace entries that are empty space with NA
test_data[test_data==""] <- NA

# Drop column if under 95% complete
df <- test_data[, colMeans(is.na(test_data)) < .05]
dim(df)

rm(final_data)
rm(test_data)

################################################################################
# Initial Data profiling report
# Missing data, QQ for quantitative and bar for qualitative
config <- list(
  "introduce" = list(),
  "plot_intro" = list(),
  "plot_str" = list(
    "type" = "diagonal",
    "fontSize" = 35,
    "width" = 1000,
    "margin" = list("left" = 350, "right" = 250)
  ),
  "plot_missing" = list(),
  "plot_qq" = list(sampled_rows = 1000L),
  "plot_bar" = list()
)
create_report(df, config = config)

################################################################################
# Examine factor variables to determine levels that cannot be used for dummy variables
df1 <- df %>% select_if(~class(.) == 'factor')
str(df1)

df <- within(df, rm(sub_grade , issue_d, title, zip_code, 
                    addr_state, earliest_cr_line, last_pymnt_d, 
                    last_credit_pull_d, policy_code))

rm(df1)

################################################################################
# Recode to binary to reduce high dimensionality due to a large number of dummy vars
df <- df %>%
  mutate(across(c(hardship_flag, debt_settlement_flag), 
                ~factor(c('N', 'Y')[(.x == "0") + 1])))

# pymnt_plan
df$pymnt_plan <- revalue(df$pymnt_plan, c("n"="0", "y"="1"))
pymnt_plan <- mapvalues(df$pymnt_plan, from = c("n", "y"),
                        to = c("0", "1"))
df$pymnt_plan <- as.numeric(df$pymnt_plan)

# initial_list_status
df$initial_list_status <- revalue(df$initial_list_status, c("f"="0", "w"="1"))
initial_list_status <- mapvalues(df$initial_list_status, from = c("f", "w"),
                                 to = c("0", "1"))
df$initial_list_status <- as.numeric(df$initial_list_status)

# application_type
df$application_type <- revalue(df$application_type, c("Individual"="0", "Joint App"="1"))
application_type <- mapvalues(df$application_type, from = c("Individual", "Joint App"),
                              to = c("0", "1"))
df$application_type <- as.numeric(df$application_type)

# Disbursement_method
df$disbursement_method <- revalue(df$disbursement_method, c("Cash"="0", "DirectPay"="1"))
disbursement_method <- mapvalues(df$disbursement_method, from = c("Cash", "DirectPay"),
                                 to = c("0", "1"))
df$disbursement_method <- as.numeric(df$disbursement_method)

################################################################################
# Final data - Look for NAs
row.has.na <- apply(df, 1, function(x){any(is.na(x))})
sum(row.has.na)
df.filtered <- df[!row.has.na,]
df <- df.filtered

rm(df.filtered)

loan_status <- df$loan_status

df <- subset(df, select = -c(loan_status))

class(loan_status)

################################################################################
# Dummy Variables using fastDummies
df$term <- as.factor(df$term)
df <- fastDummies::dummy_cols(df)

# Remove original vars dummy vars were made from
df <- within(df, rm(term, grade, emp_length, home_ownership, verification_status,
                    purpose, hardship_flag, debt_settlement_flag))

df <- cbind(loan_status, df)
summary(df)

################################################################################

# Semi-clean Data profiling report
# Missing data, QQ for quantitative and bar for qualitative
config <- list(
  "introduce" = list(),
  "plot_intro" = list(),
  "plot_str" = list(
    "type" = "diagonal",
    "fontSize" = 35,
    "width" = 1000,
    "margin" = list("left" = 350, "right" = 250)
  ),
  "plot_missing" = list(),
  "plot_qq" = list(sampled_rows = 1000L),
  "plot_bar" = list()
)
create_report(df, config = config)

################################################################################
# Examine dependent variable
levels(df$loan_status)
table <- table(df$loan_status)
round(prop.table(table))

# re-order levels
reorder_size <- function(x) {
  factor(x, levels = names(sort(table(x), decreasing = TRUE)))
}

ggplot(df, aes(x = reorder_size(`loan_status`))) +
  geom_bar() +
  xlab("Loan Status") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Modeling Set Up for Binary Classification 
# Dependent variable = loan_status 
# Change loan_status to a binary response, either Current or Default

# a.	Current = all responses that suggest the loan is on time and being paid 
# Fully Paid, Grace Period, Current

# b.	Default = all responses that suggest the loan is late or charged off 
# The remaining responses:
# Charged Off
# Does not meet the credit policy. Status:Charged Off = 0 so could drop
# Does not meet the credit policy. Status:Fully Paid = 0 so could drop
# Late (16-30 days)
# Late (31-120 days)

# View percentages of groups in dependent variable
table <- table(df$loan_status)
prop.table(table)

# Recode current
df$loan_status<- revalue(df$loan_status, c("Fully Paid"="Current", "In Grace Period"="Current"))
loan_status <- mapvalues(df$loan_status, from = c("Fully Paid", "In Grade Period"),
                         to = c("Current", "Current"))

# Recode default
df$loan_status<- revalue(df$loan_status, c("Charged Off"="Default",
                                             "Does not meet the credit policy. Status:Charged Off"="Default",
                                             "Does not meet the credit policy. Status:Fully Paid"="Default",
                                             "Late (16-30 days)"="Default",
                                             "Late (31-120 days)"="Default"))
loan_status <- mapvalues(df$loan_status, from = c("Charged Off", "Does not meet the credit policy. Status:Charged Off",
                                                   "Does not meet the credit policy. Status:Fully Paid",
                                                   "Late (16-30 days)",
                                                   "Late (31-120 days)"),
                         to = c("Default","Default","Default","Default","Default"))

#Recode current to 0, default 1
df$loan_status<- revalue(df$loan_status, c("Current"="1", "Default"="0"))
loan_status <- mapvalues(df$loan_status, from = c("Current", "Default"),
                         to = c("1", "0"))

table <- table(df$loan_status)
prop.table(table)

# Write data to csv
write.csv(df, 'LoanStatus_withClassImbalancce.csv')

################################################################################
# Convert Loan_status column from Default/Current to 1/0
class(df$loan_status) #factor
Loan_status = as.numeric(as.factor(df$loan_status))-1 
df = cbind(Loan_status,df) 
df <- within(df, rm(loan_status))

################################################################################
############################## Variable Selection ##############################
################################################################################
# Partition for variable selection methods
set.seed(123)
data_sampling_vector <- createDataPartition(df$Loan_status, p=0.99, list = FALSE)

data_train <- df[data_sampling_vector,]
data_test <- df[-data_sampling_vector,]

################################################################################
################################################################################
# MVSIS - model free screening for variable selection - source code for optimization

a <- data_test
dim(a)

write.table(a,"names.csv",sep=",", col.names=T,row.names=F)

#Write data frame to txt to test optimization screening
write(t(a),"LoanStatus_screen.txt",ncol=ncol(a))

################################################################################
# MVSIS - R package
df.mvsis <- data_test
dim(df.mvsis)

df.mvsis$Loan_status <- as.integer(df.mvsis$Loan_status)

# Set up for screening
X <- df.mvsis[,2:112]
Y <- df.mvsis[,1]

# Screen using MV-SIS
A <- screenIID(X, Y, method = "MV-SIS")

l1= list(A[1])
l2= list(A[2])

lapply(l1, function(x) write.table(data.frame(x), 'MVSIS.measure.csv',
                                    append= T, sep=',' ))
lapply(l2, function(x) write.table(data.frame(x), 'MVSIS.rank.csv',
                                    append= T, sep=',' ))

################################################################################
################################################################################
# Boruta for variable selection
EnsurePackage<-function(x)
{
  x<-as.character(x)
  if (!require(x,character.only=TRUE))
  {
    install.packages(pkgs=x,dependencies = TRUE)
    require(x,character.only=TRUE)
  }
} 

EnsurePackage("Boruta")
library(Boruta)
set.seed(123)

# Response is a factor for classification
data_test$Loan_status <- as.factor(data_test$Loan_status)

# Model
boruta.df <- Boruta(Loan_status~., data = data_test, doTrace = 2)
print(boruta.df) 

# Results
# Get the variable importance chart
plot(boruta.df, xlab = "", xaxt = "n")

lz<-lapply(1:ncol(boruta.df$ImpHistory),function(i)
  boruta.df$ImpHistory[is.finite(boruta.df$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.df$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.df$ImpHistory), cex.axis = 0.7) 

# The green color box plots indicate important variables
# The red color indicates unimportant 
# The yellow color indicates tentative variables.

# Explore tentative variables
final.boruta <- TentativeRoughFix(boruta.df)
print(final.boruta) 

# Now plot the variable importance chart again
plot(final.boruta, xlab = "", xaxt = "n")

lz<-lapply(1:ncol(final.boruta$ImpHistory),function(i)
  final.boruta$ImpHistory[is.finite(final.boruta$ImpHistory[,i]),i])
names(lz) <- colnames(final.boruta$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(final.boruta$ImpHistory), cex.axis = 0.7) 

# Print all the important attributes
getSelectedAttributes(final.boruta, withTentative = F) 

# Additional statistical measures
boruta.df <- attStats(final.boruta)
print(boruta.df) 

################################################################################
