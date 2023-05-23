import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split,KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

train_df=pd.read_pickle('./preprocessed_train.pkl')
y=train_df['SalePrice']
x=train_df.drop(['SalePrice'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=36)

def stacking_base_datasets(model,x_train_n,y_train_n,x_test_n,n_folds):
    kf=KFold(n_splits=n_folds,shuffle=True,random_state=36)

    #메타 모델이 학습할 데이터 반환을 위해 넘파이 배열 초기화
    train_fold_pred=np.zeros((x_train_n.shape[0],1))
    test_pred=np.zeros((x_test_n.shape[0],n_folds))
    print(model.__class__.__name__,'모델 시작')

    for folder_cnt, (train_index,valid_index) in enumerate(kf.split(x_train_n)):
        print('\t fold set:', folder_cnt, 'start')
        x_tr=x_train_n[train_index]
        y_tr=y_train_n[train_index]
        x_te=x_train_n[valid_index]

        model.fit(x_tr,y_tr)
        train_fold_pred[valid_index,:]=model.predict(x_te).reshape(-1,1)
        test_pred[:,folder_cnt]=model.predict(x_test_n)

        test_pred_mean=np.mean(test_pred,axis=1).reshape(-1,1)

    return train_fold_pred, test_pred_mean


x_train_n=x_train.values
y_train_n=y_train.values
x_test_n=x_test.values


ridge=Ridge(alpha=10)
lasso=Lasso(alpha=0.001)
xgb=XGBRegressor(n_estimators=1000,learning_rate=0.05,colsample_bytree=0.5,subsample=0.8)
lgbm=LGBMRegressor(n_estimators=1000,learning_rate=0.05,num_leaves=4,colsample_bytree=0.4,subsample=0.6,reg_lambda=10,n_jobs=-1)

ridge_train,ridge_test=stacking_base_datasets(ridge,x_train_n,y_train_n,x_test_n,5)
lasso_train, lasso_test = stacking_base_datasets(lasso, x_train_n, y_train_n, x_test_n, 5)
xgb_train, xgb_test = stacking_base_datasets(xgb, x_train_n, y_train_n, x_test_n, 5)
lgbm_train, lgbm_test = stacking_base_datasets(lgbm, x_train_n, y_train_n, x_test_n, 5)

stack_x_train=np.concatenate((ridge_train,lasso_train,xgb_train,lgbm_train),axis=1)
stack_x_test = np.concatenate((ridge_test, lasso_test, xgb_test, lgbm_test), axis=1)

meta_model=Lasso(alpha=0.0005)

meta_model.fit(stack_x_train,y_train)
meta_pred=meta_model.predict(stack_x_test)
rmse=mean_squared_error(y_test,meta_pred,squared=False)
print("Stacking final meta model's RMSE :", rmse)