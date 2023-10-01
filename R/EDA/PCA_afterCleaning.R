################################################################################
############ Principal Component Analysis after Cleaning Data ##################
################################################################################
library(tidyverse)
library(factoextra)

valid_column_names <- make.names(names=names(df), unique=TRUE, allow_=TRUE)
names(df) <- valid_column_names

names_freq <- as.data.frame(table(names(df)))
names_freq[df$Freq > 1,]

# Create a dataframe with only numeric values for testing normality
df_num <- select_if(df, is.numeric)

# PCA
df <- df[,which(apply(df, 2, var) != 0)]
df.prcomp <- prcomp(df, center=TRUE, scale.=TRUE)
summary(df.prcomp)

screeplot(df.prcomp, type="l", npcs=30, main="Screeplot of the first 30 PCs")
abline(h=1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue=1"),
       col=c("red"), lty=5, cex=0.6)

cumpro <- cumsum(df.prcomp$sdev^2 / sum(df.prcomp$sdev^2))
plot(cumpro[0:66], xlab="PC #", ylab="Amount of explained variance", main="Cumulative variance plot")
abline(v=65, col="blue", lty=5)
abline(h=0.9, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC65"),
       col=c("blue"), lty=5, cex=0.6)

fviz_pca_var(df.prcomp,
             col.var="contrib", # Color by contributions to the PC
             gradient.cols=c("#00AFBB", "#E7B800", "#FC4E07"),
             repel=TRUE     # Prevent text from overlapping
)

# Eigenvalues
eig.val <- get_eigenvalue(df.prcomp)
eig.val

# Results for Variables
res.var <- get_pca_var(df.prcomp)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 