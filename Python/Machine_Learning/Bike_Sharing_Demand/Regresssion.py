import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
import os
warnings.filterwarnings('ignore',category=RuntimeWarning)

#print(os.getcwd())
bike_df=pd.read_csv('./train.csv')
print('Data Shape')
print(bike_df.shape,end='\n')

print('Show data 5 rows')
print(bike_df.head(),end='\n')

print('Show data information')
print(bike_df.info(),end='\n')

#[datetime] : 문자열 => datetime
bike_df['datetime']=bike_df.datetime.apply(pd.to_datetime)

#datetime 타입에서 년,월,일 시간 추출
bike_df['year']=bike_df.datetime.apply(lambda x : x.year)
bike_df['month']=bike_df.datetime.apply(lambda x : x.month)
bike_df['day']=bike_df.datetime.apply(lambda x : x.day)
bike_df['hour']=bike_df.datetime.apply(lambda x : x.hour)
print('Create time features')
print(bike_df.head(),end='\n')

#casual+registered=count , 상관관계가 높으므로 삭제, datetime도 삭제
drop_columns=['datetime','casual','registered']
bike_df.drop(drop_columns,axis=1,inplace=True)
print('Delete datetime,casual,registered features')
print(bike_df.head(),end='\n')

'''
RMSLE(Root Mean Square Log Error)
log 값 변환 시 NaN 등의 이슈로 log()가 아닌 log1p()를 이용해 RMSLE 계산
log1p()=1+log()
그냥 log()를 사용하면 overflow/underflow 발생하기 쉬움
expm1()사용시 log1p() 다시 변롼 가
'''
def rmsle(y,pred):
    log_y=np.log1p(y)
    log_pred=np.log1p(pred)
    squared_error=(log_y-log_pred)**2
    rmsle=np.sqrt(np.mean(squared_error))
    return rmsle


def evaluate_regr(y,pred):
    rmsle_val=rmsle(y,pred)
    rmse_val=mean_squared_error(y,pred,squared=False)
    mae_val=mean_absolute_error(y,pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3f}, MAE:{2:.3f}'.format(rmsle_val,rmse_val,mae_val))

#Model Training
y=bike_df['count']
x_features=bike_df.drop(['count'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x_features,y,test_size=0.3,random_state=36)

lr_reg=LinearRegression()
lr_reg.fit(x_train,y_train)
pred=lr_reg.predict(x_test)

print('Linear result with no scaled y')
evaluate_regr(y_test,pred)
print(end='\n\n')


#error가 가장 큰 순으로 5개 check
def get_top_error_data(y_test,pred,n_tops=5):
    result_df=pd.DataFrame(y_test.values,columns=['real_count'])
    result_df['pred_count']=np.round(pred)
    result_df['diff']=np.abs(result_df['real_count']-result_df['pred_count'])

    print(result_df.sort_values('diff',ascending=False)[:n_tops])

print('Top5 error data')
get_top_error_data(y_test,pred,n_tops=5)
print(end='\n\n')

#y true 값 분포 check
plt.figure(1)
y.hist()
plt.title('y distribution')

#log1p 스케일 변환 후 분포 check
plt.figure(2)
y_log_transform=np.log1p(y)
y_log_transform.hist()
plt.title('log1p scaled y distribution')


#y값에 scale 적용
y_log=np.log1p(y)
x_train,x_test,y_train,y_test=train_test_split(x_features,y_log,test_size=0.3,random_state=36)

lr_reg=LinearRegression()
lr_reg.fit(x_train,y_train)
pred=lr_reg.predict(x_test)

#로그 변환이 되어있으므로 expm1을 사용해서 원래 스케일로 변환
y_test_exp=np.expm1(y_test)
pred_exp=np.expm1(pred)

print('Linear result with log1p scaled y')
evaluate_regr(y_test_exp,pred_exp)
print(end='\n\n')

#feature coef check
coef=pd.Series(lr_reg.coef_,index=x_features.columns)
coef_sort=coef.sort_values(ascending=False)
plt.figure(3)
sns.barplot(x=coef_sort.values,y=coef_sort.index)
plt.title('linear regression coef with log1p scaled y')


#year,month,day,hour,holiday,workingday,season,weather : numeric => category 변환
x_features_ohe=pd.get_dummies(x_features,columns=['year','month','day','hour','holiday','workingday','season','weather'])

x_train,x_test,y_train,y_test=train_test_split(x_features_ohe,y_log,test_size=0.3,random_state=36)

def get_model_predict(model,x_train,x_test,y_train,y_test,is_expm1=False):
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    if is_expm1:
        y_test=np.expm1(y_test)
        pred=np.expm1(pred)
    print('###',model.__class__.__name__,'###')
    evaluate_regr(y_test,pred)

#모델 별로 평가
lr_reg=LinearRegression()
ridge_reg=Ridge(alpha=10)
lasso_reg=Lasso(alpha=0.01)

print('Linear Models result with log1p scaled y and categorical x')
for model in [lr_reg,ridge_reg,lasso_reg]:
    get_model_predict(model,x_train,x_test,y_train,y_test,is_expm1=True)
print(end='\n\n')

#feature coef check
coef=pd.Series(lr_reg.coef_,index=x_features_ohe.columns)
coef_sort=coef.sort_values(ascending=False)[:20]
plt.figure(4)
sns.barplot(x=coef_sort.values,y=coef_sort.index)
plt.title('linear regression coef, scaled y, category x')


rf_reg=RandomForestRegressor(n_estimators=500)
gbm_reg=GradientBoostingRegressor(n_estimators=500)
xgb_reg=XGBRegressor(n_estimators=500)
lgbm_reg=LGBMRegressor(n_estimators=500)

#xgboost의 경우 dataframe 입력 될 경우 버전에 따라 오류 발생 가능 , ndarray로 변환
print("RF, GBR, XGB, LGBM result with log1p scaled y and categorical x")
for model in [rf_reg, gbm_reg, xgb_reg,lgbm_reg]:
    get_model_predict(model, x_train.values, x_test.values, y_train.values, y_test.values, is_expm1 = True)

plt.show()



