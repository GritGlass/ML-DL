import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import skew
import os

#print(os.getcwd())

#Data Check
house_df_org=pd.read_csv('./train.csv')
house_df=house_df_org.copy()
print('Raw data')
print(house_df.head(),end='\n\n')
print('데이터 set의 shape:', house_df.shape)
print('\n전체 feature의 type \n',house_df.dtypes.value_counts())
isnull_series=house_df.isnull().sum()
print('\nNull column과 해당 개수:\n', isnull_series[isnull_series>0].sort_values(ascending=False),end='\n\n')

#Null이 많은 PoolQC,MiscFeature,Alley,Fence 삭제
house_df.drop(['PoolQC','MiscFeature','Alley','Fence','Id'],axis=1,inplace=True)


plt.figure(1)
plt.title('Original Sale Price Histogram')
sns.distplot(house_df['SalePrice'])

plt.figure(2)
plt.title('Log Transformed sale price histogram')
log_saleprice=np.log1p(house_df['SalePrice'])
sns.distplot(log_saleprice)

#SalePrice log1p 변환
house_df['SalePrice']=log_saleprice
#숫자형 null 컬럼은 평균으로 대체
house_df.fillna(house_df.mean(),inplace=True)

#NUll이 있는 feature명과 type
null_col_cnt=house_df.isnull().sum()[house_df.isnull().sum()>0]
print("@Null feature's type:\n",house_df.dtypes[null_col_cnt.index],end='\n\n')

#dummy 만들기
print('get_dummies() 수행 전 데이터 shape:',house_df.shape)
house_df_ohe=pd.get_dummies(house_df)
print('get_dummies() 수행 후 데이터 shape :', house_df_ohe.shape,end='\n\n')
#NUll check again
null_col_cnt=house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum()>0]
print("@Null feature's type:\n",house_df_ohe.dtypes[null_col_cnt.index],end='\n\n')

#error 계산 함수
def get_rmse(model):
    pred=model.predict(x_test)
    rmse=mean_squared_error(y_test,pred,squared=False)
    print(model.__class__.__name__,'RMSE: ',rmse)
    return np.round(rmse,3)
def get_rmses(models):
    rmses=[]
    for model in models:
        rmse=get_rmse(model)
        rmses.append(rmse)
    return rmses

#Model Train
y=house_df_ohe['SalePrice']
x_features=house_df_ohe.drop('SalePrice',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x_features,y,test_size=0.2,random_state=36)

lr=LinearRegression()
lr.fit(x_train,y_train)
ridge=Ridge()
ridge.fit(x_train,y_train)
lasso=Lasso()
lasso.fit(x_train,y_train)

models=[lr,ridge,lasso]
get_rmses(models)
print('\n')

#회귀 계수의 값 상위 10개, 하위 10개 추출
def top_bottom_coef(model,n=10):
    coef=pd.Series(model.coef_,index=x_features.columns)
    coef_high=coef.sort_values(ascending=False).head(n)
    coef_low=coef.sort_values(ascending=True).head(n)
    return coef_high,coef_low

def visual_coef(models):
    #3개의 모델을 시각화 하기 위해서 subplot 생성
    fig,axs=plt.subplots(figsize=(20,6),nrows=1,ncols=3)
    fig.tight_layout()

    for i, model in enumerate(models):
        #상위,하위 각각 10개 회귀 계수 concat으로 결합
        coef_high,coef_low=top_bottom_coef(model)
        coef_concat=pd.concat([coef_high,coef_low])

        axs[i].set_title(model.__class__.__name__+' Coefficients',size=20)
        axs[i].tick_params(axis='y',direction='in',pad= -120)

        for label in (axs[i].get_xticklabels()+axs[i].get_yticklabels()):
            label.set_fontsize(15)
        sns.barplot(x=coef_concat.values,y=coef_concat.index,ax=axs[i])


visual_coef(models)

def avg_rmse_cv(models):
    for model in models:
        rmse_list=-cross_val_score(model,x_features,y,scoring='neg_root_mean_squared_error',cv=5)
        rmse_avg=np.mean(rmse_list)
        print('{0} CV RMSE 값 리스트: {1}'.format(model.__class__.__name__,np.round(rmse_list,3)))
        print('{0} CV 평균 RMSE 값 : {1}\n'.format(model.__class__.__name__, np.round(rmse_avg, 3)))

models=[lr,ridge,lasso]
avg_rmse_cv(models)

#최적 alpha값 찾기
def print_best_params(model,params):
    grid_model=GridSearchCV(model,param_grid=params,scoring='neg_root_mean_squared_error',cv=5)
    grid_model.fit(x_features,y)
    rmse=-1*grid_model.best_score_
    print('{0} 5 CV 시 최적 평균 RMSE 값 : {1}, 최적 alpha : {2}\n'.format(model.__class__.__name__,np.round(rmse,4),grid_model.best_params_))

# ridge_params={'alpha':[0.05,0.1,1,5,8,10,12,15,20]}
# lasso_params={'alpha':[0.001,0.005,0.008,0.05,0.03,0.1,0.5,1,5,10]}
# print_best_params(ridge,ridge_params)
# print_best_params(lasso,lasso_params)
#
# #Best param으로 재학습 및 평가
# print('Best Param result')
# ridge=Ridge(alpha=12)
# ridge.fit(x_train,y_train)
# lasso=Lasso(alpha=0.001)
# lasso.fit(x_train,y_train)
#
# models=[lr,ridge,lasso]
# get_rmses(models)
# visual_coef(models)
# print('\n\n')

#---------------------------------------------------------------------------------------------------------------
#house_df의 왜곡된 정도 check

#object가 아닌 숫자형 feature의 colimn index 추출
features_index=house_df.dtypes[house_df.dtypes != 'object'].index
skew_features=house_df[features_index].apply(lambda x: skew(x))
#왜곡이 큰 feature만 추출
skew_features_top=skew_features[skew_features>1]
print('왜곡이 큰 features \n',skew_features_top.sort_values(ascending=False),end='\n\n')

#왜곡이 큰 feature 로그 변환
house_df[skew_features_top.index]=np.log1p(house_df[skew_features_top.index])

#다시 더미 변수 생성 모델 재학습
house_df_ohe=pd.get_dummies(house_df)
# y=house_df_ohe['SalePrice']
# x_features=house_df_ohe.drop(['SalePrice'],axis=1)
# x_train,x_test,y_train,y_test=train_test_split(x_features,y,test_size=0.2,random_state=36)
#
# ridge_params={'alpha':[0.05,0.1,1,5,8,10,12,15,20]}
# lasso_params={'alpha':[0.001,0.005,0.008,0.05,0.03,0.1,0.5,1,5,10]}
# print_best_params(ridge,ridge_params)
# print_best_params(lasso,lasso_params)
#
# #Best param으로 재학습 및 평가
# print('Best Param result')
# ridge=Ridge(alpha=10)
# ridge.fit(x_train,y_train)
# lasso=Lasso(alpha=0.001)
# lasso.fit(x_train,y_train)
#
# models=[lr,ridge,lasso]
# get_rmses(models)
# visual_coef(models)
# print('\n\n')

#------------------------------------------------------------------------------------------------------------------
#outlier 제거
#원본 데이터에서 saleprice와 grlivarea check
plt.figure(4)
plt.scatter(x=house_df_org['GrLivArea'],y=house_df_org['SalePrice'])
plt.ylabel('SalePrice',fontsize=15)
plt.xlabel('GrLivArea',fontsize=15)

#조건
cond1=house_df_ohe['GrLivArea']>np.log1p(4000)
cond2=house_df_ohe['SalePrice']>np.log1p(500000)
outlier_index=house_df_ohe[cond1 & cond2].index

print('이상치 개수 : ', outlier_index.values)
print('이상치 제거 전 : ', house_df_ohe.shape)
house_df_ohe.drop(outlier_index,axis=0,inplace=True)
print('이상치 제거 후 : ', house_df_ohe.shape,end='\n\n')

#재학
y=house_df_ohe['SalePrice']
x_features=house_df_ohe.drop(['SalePrice'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x_features,y,test_size=0.2,random_state=36)

ridge_params={'alpha':[0.05,0.1,1,5,8,10,12,15,20]}
lasso_params={'alpha':[0.001,0.005,0.008,0.05,0.03,0.1,0.5,1,5,10]}
print_best_params(ridge,ridge_params)
print_best_params(lasso,lasso_params)

#Best param으로 재학습 및 평가습
print('Best Param result')
lr=LinearRegression()
lr.fit(x_train,y_train)
ridge=Ridge(alpha=10)
ridge.fit(x_train,y_train)
lasso=Lasso(alpha=0.001)
lasso.fit(x_train,y_train)

models=[lr,ridge,lasso]
get_rmses(models)
visual_coef(models)
print('\n\n')

#XGB
plt.figure(6)
#xgb_params={'n_estimators':[1000]}
xgb=XGBRegressor(n_estimators=1000,learning_rate=0.05,colsample_bytree=0.5,subsample=0.8)
xgb.fit(x_train,y_train)
#print_best_params(xgb,xgb_params)

xgb_import=pd.Series(xgb.feature_importances_,index=xgb.feature_names_in_)
xgb_top=xgb_import.sort_values(ascending=False)[:20]
plt.title('XGB feature Importance')
sns.barplot(x=xgb_top.values,y=xgb_top.index)

#LGBM
plt.figure(7)
#lgbm_params={'n_estimators':[1000]}
lgbm=LGBMRegressor(n_estimators=1000,learning_rate=0.05,num_leaves=4,colsample_bytree=0.4,subsample=0.6,reg_lambda=10,n_jobs=-1)
lgbm.fit(x_train,y_train)
#print_best_params(lgbm,lgbm_params)

lgbm_import=pd.Series(lgbm.feature_importances_,index=lgbm.feature_name_)
lgbm_top=lgbm_import.sort_values(ascending=False)[:20]
plt.title('LGBM feature Importance')
sns.barplot(x=lgbm_top.values,y=lgbm_top.index)

models=[xgb,lgbm]
get_rmses(models)


#----------------------------------------------------------------------------------------------------------------------
#Model 혼합

def rmse_pred(preds):
    for key in preds.keys():
        pred_value=preds[key]
        rmse=np.round(mean_squared_error(y_test,pred_value,squared=False),3)
        print('{0} 모델의 RMSE: {1}'.format(key,rmse))

ridge_pred=ridge.predict(x_test)
lasso_pred=lasso.predict(x_test)
xgb_pred=xgb.predict(x_test)
lgbm_pred=lgbm.predict(x_test)

ridge_lasso_pred=0.4*ridge_pred+0.6*lasso_pred
xgb_lgbm_pred=0.5*xgb_pred+0.5*lgbm_pred

preds={'Ridge_Lasso':ridge_lasso_pred,
       'Ridge':ridge_pred,
       'Lasso':lasso_pred,
       'XGB_LGBM':xgb_lgbm_pred,
       'XGB':xgb_pred,
       'LGBM':lgbm_pred}

print('\n\n')
print('MIX MODEL RESULT')
rmse_pred(preds)
print('\n\n')


plt.show()

house_df_ohe.to_pickle('./preprocessed_train.pkl')
