library(readxl)
Dataset <- read_excel("fraud_data_2024.xlsx")
head(Dataset)
str(Dataset)
Dataset$category <- as.factor(Dataset$category)
Dataset$state <- as.factor(Dataset$state)
Dataset$job <- as.factor(Dataset$job)
Dataset$is_fraud <- factor(Dataset$is_fraud,levels = c(0,1))

str(Dataset)
colnames(Dataset) <- c("date","merchant","category","amt",
                       "city","state","lat","long","city_pop","job","dob","trans_num",
                       "merch_lat","merch_long","is_fraud")
#### Part B ####
### Soal 1 ###

# Uji Multikolinieritas
check_model <- lm(amt ~ state + city_pop + is_fraud ,data=Dataset)
summary(check_model)
library(car)
vif(check_model)

# Model Common Effect
library(plm)
cem <- plm(amt ~ state + city_pop + is_fraud,data=Dataset, model = "pooling")
summary(cem)

# Model Fixed Effect
fem <- plm(amt ~ state + city_pop + is_fraud ,data=Dataset, index = c("city", "date"), model = "within", effect= "individual")
summary(fem)
summary(fixef(fem, effect="individual"))

# Model FEM dengan waktu
fem_time <- plm(amt ~ state + city_pop + is_fraud ,data=Dataset, index = c("city", "date"), model = "within", effect= "time")
summary(fem_time)
summary(fixef(fem_time, effect="time"))

# Nilai Kebaikan Model
# Sum Squared Error
dsse <- data.frame(Individu=sum(fem$residuals^2),Time=sum(fem_time$residuals^2))
drsq <- data.frame(Individu=summary(fem)$r.squared[1],Time=summary(fem_time)$r.squared[1])

# MAPE
mape <- function(actual, forecast) {
  mean(abs((actual - forecast) / actual)) * 100
}
dmape <- data.frame(Individu=mape(Dataset$amt,predict(fem)),
                    Time=mape(Dataset$amt,predict(fem_time)))

# Perbandingan
compare <- t(rbind(dsse,drsq,dmape))
colnames(compare) <- c("SSE","R-Squared", "MAPE")
compare

# FEM VS CEM
pooltest(cem, fem) # Keputusan pilih FEM

# REM dengan Generalized Least Square
rem_gls <- plm(amt ~ state + city_pop + is_fraud, data = Dataset, 
               index = c("city", "date"), 
               effect = "individual", model = "random", random.method = "nerlove")
summary(rem_gls)

#efek individu
plmtest(rem_gls,type = "bp", effect="individu")

#efek waktu 
plmtest(rem_gls,type = "bp", effect="time")

#efek twoways 
plmtest(rem_gls,type = "bp", effect="twoways")

# FEM VS REM
# Uji Haussman
phtest(fem, rem_gls) # Pilih model REM

# Uji diagnostik residu
# Uji Autokorelasi
pbgtest(rem_gls)

# Uji Heteroskedastisitas
library(lmtest)
bptest(rem_gls)

# Check juga untuk model linear biasa
# Uji Autokorelasi
dwtest(check_model)

# Uji Heteroskedastisitas
bptest(check_model)

# Model linear dengan hanya data numerik
model.numeric <- lm(amt ~ city_pop ,data=Dataset)
summary(model.numeric)
dwtest(model.numeric) # Uji Autokorelasi
bptest(model.numeric) # Uji Heteroskedastisitas

### Soal 2 ###
library(tidyverse) 
library(corrplot)
library(gridExtra)
library(GGally)
library(cluster) 
library(factoextra) 

Data2 <- Dataset[,c(4,5,6,7,8,9,13,14)]
corrplot(cor(Data2[,-c(2,3)]), type = 'upper', method = 'number', tl.cex = 0.9)
Data.New <- Data2[,-c(4,5,7,8)]

library(dplyr)
Norm <- as.data.frame(scale(Data.New[,-c(2,3)]))
head(Norm)

set.seed(123)
k.means.model <- kmeans(Norm, centers = 3)
print(k.means.model)

fviz_cluster(k.means.model, data = Norm)

k.means.model$cluster # Clusters to which each point is associated
k.means.model$centers # Cluster centers
k.means.model$size # Cluster size
k.means.model$betweenss # Between clusters sum of square
k.means.model$withinss # Within cluster sum of square
k.means.model$tot.withinss # Total with sum of square
k.means.model$totss # Total sum of square

Data.New[,-c(2,3)] %>% 
  mutate(Cluster = k.means.model$cluster) %>%
  group_by(Cluster) %>%
  summarize_all('median')

df.cluster = data.frame(Data.New, k.means.model$cluster)
most.size <- df.cluster[which(df.cluster$k.means.model.cluster == 1),]
table(most.size$state)

### Soal 3 ###
set.seed(123)

# Set the proportion of data to be used for training
train_proportion <- 0.8

# Determine the number of samples for training
num_train_samples <- round(nrow(Dataset) * train_proportion)

# Randomly sample row indices for the training set
train_indices <- sample(seq_len(nrow(Dataset)), size = num_train_samples, replace = FALSE)

# Create the training set
train_set <- Dataset[train_indices, ]

# Create the testing set by excluding the training indices
test_set <- Dataset[-train_indices, ]

dim(train_set) 
dim(test_set) 

library(e1071)
#Pembuatan Model NaiveBayes 
model_naive <- naiveBayes(is_fraud ~ amt + state + city_pop + job , data = train_set) 

#Prediksi kelas target pada dataset validasi (topredict)
preds_naive <- predict(model_naive, newdata = test_set[,c(4,6,9,10)])  
conf_matrix_naive <- table(preds_naive, test_set$is_fraud)
conf_matrix_naive
library(caret)
confusionMatrix(conf_matrix_naive)  



