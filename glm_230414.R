
df_x_230413 <- read.csv('승하차_일반인_컬럼작업완_230414.csv', fileEncoding = 'cp949')
df_x_230413 <- df_x_230413[-1]

df1 <- read.csv('grid_total.csv', encoding='UTF-8')
df1 <- df1[-1]


df_x_230413$등급 <- df1$등급
df_x_sc$등급 <- df1$등급
df_x_sc$등급 <- factor(df1$등급)
df_x <- df_x_230413[c(2,3,4,6,10)]
df_x_sc <- log1p(df_x)

model_lm <- lm(등급 ~ 퇴근시간대+대합실면적+스타벅스수+퇴근시간대승차인원+퇴근시간대일반인승차, data = df_x_sc)
model_glm <- glm(등급 ~ 퇴근시간대+대합실면적+스타벅스수+퇴근시간대승차인원+퇴근시간대일반인승차, data = df_x_sc,family = binomial())
# Call:
#   glm(formula = 등급 ~ 퇴근시간대 + 대합실면적 + 스타벅스수 + 
#         퇴근시간대승차인원 + 퇴근시간대일반인승차, 
#       data = df_x_sc)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -0.9794  -0.2894   0.0614   0.2993   0.9596  
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)           4.83898    0.88696   5.456 1.47e-07 ***
#   퇴근시간대            0.01916    0.12021   0.159    0.874    
# 대합실면적           -0.38475    0.07764  -4.956 1.56e-06 ***
#   스타벅스수           -0.24649    0.05260  -4.686 5.21e-06 ***
#   퇴근시간대승차인원    0.55333    0.42130   1.313    0.191    
# 퇴근시간대일반인승차 -0.54855    0.44385  -1.236    0.218    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for gaussian family taken to be 0.1668282)
# 
# Null deviance: 45.313  on 200  degrees of freedom
# Residual deviance: 32.531  on 195  degrees of freedom
# AIC: 218.37
# 
# Number of Fisher Scoring iterations: 2

summary(model_glm)
# Call:
#   glm(formula = 등급 ~ 퇴근시간대 + 대합실면적 + 스타벅스수 + 
#         퇴근시간대승차인원 + 퇴근시간대일반인승차, 
#       family = binomial(), data = df_x_sc)
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.3520  -0.7175   0.4044   0.7511   2.3258  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)           18.7825     5.6791   3.307 0.000942 ***
#   퇴근시간대             0.1145     0.7115   0.161 0.872117    
# 대합실면적            -2.1876     0.5038  -4.342 1.41e-05 ***
#   스타벅스수            -1.3988     0.3496  -4.001 6.30e-05 ***
#   퇴근시간대승차인원     3.4391     2.5543   1.346 0.178170    
# 퇴근시간대일반인승차  -3.4041     2.6723  -1.274 0.202709    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 258.56  on 200  degrees of freedom
# Residual deviance: 195.72  on 195  degrees of freedom
# AIC: 207.72
# 
# Number of Fisher Scoring iterations: 4


# ------------------------------------------------------------------------------
# 필요한 패키지 로딩
library(dplyr)
library(caret)
library(glmnet)
df_x_sc
# train/test 데이터셋 분리
set.seed(123)
train_index <- createDataPartition(df_x_sc$등급, p = 0.7, list = FALSE)
train <- df_x_sc[train_index,]
test <- df_x_sc[-train_index,]

# k-fold 교차검증을 위한 제어판 생성
set.seed(123)
fitControl <- trainControl(method = "cv", number = 4)

# 로지스틱 회귀모델 학습
fit <- train(등급 ~ ., data = train, method = "glmnet", trControl = fitControl)

# 모델 성능 평가
pred <- predict(fit, newdata = test)
confusionMatrix(pred, test$등급)

# Confusion Matrix and Statistics
# Reference
# Prediction  1  2
#          1 10  6
#          2 10 33
# 
# Accuracy : 0.7288          
# 95% CI : (0.5973, 0.8364)
# No Information Rate : 0.661           
# P-Value [Acc > NIR] : 0.1681          
# 
# Kappa : 0.3639          
# 
# Mcnemar's Test P-Value : 0.4533          
#                                           
#             Sensitivity : 0.5000          
#             Specificity : 0.8462          
#          Pos Pred Value : 0.6250          
#          Neg Pred Value : 0.7674          
#              Prevalence : 0.3390          
#          Detection Rate : 0.1695          
#    Detection Prevalence : 0.2712          
#       Balanced Accuracy : 0.6731          
#                                           
#        'Positive' Class : 1



