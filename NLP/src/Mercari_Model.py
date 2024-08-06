import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from lightgbm import LGBMRegressor
from tabulate import tabulate
from scipy.sparse import hstack
import gc
import warnings
warnings.filterwarnings('ignore')

mercari_df=pd.read_csv('train.tsv', sep='\t')
def split_cat(category_name):
    try:
        return category_name.split('/')
    except:
        return ['Other_Null','Other_Null','Other_Null']

def preprocessing(mercari_df):
    mercari_df['cat_dae'], mercari_df['cat_jung'], mercari_df['cat_so'] = \
        zip(*mercari_df['category_name'].apply(lambda x: split_cat(x)))
    mercari_df['brand_name'] = mercari_df['brand_name'].fillna(value='Other_Null')
    mercari_df['category_name'] = mercari_df['category_name'].fillna(value='Other_Null')
    mercari_df['item_description'] = mercari_df['item_description'].fillna(value='Other_Null')
    cnt_vec = CountVectorizer()
    x_name = cnt_vec.fit_transform(mercari_df.name)

    tfidf_descp = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), stop_words='english')
    x_descp = tfidf_descp.fit_transform(mercari_df['item_description'])

    lb_brand_name = LabelBinarizer(sparse_output=True)
    x_brand = lb_brand_name.fit_transform(mercari_df['brand_name'])
    lb_item_cond_id = LabelBinarizer(sparse_output=True)
    x_item_cond_id = lb_item_cond_id.fit_transform(mercari_df['item_condition_id'])
    lb_shipping = LabelBinarizer(sparse_output=True)
    x_shipping = lb_shipping.fit_transform(mercari_df['shipping'])

    lb_cat_dae = LabelBinarizer(sparse_output=True)
    x_cat_dae = lb_cat_dae.fit_transform(mercari_df['cat_dae'])
    lb_cat_jung = LabelBinarizer(sparse_output=True)
    x_cat_jung = lb_cat_jung.fit_transform(mercari_df['cat_jung'])
    lb_cat_so = LabelBinarizer(sparse_output=True)
    x_cat_so = lb_cat_so.fit_transform(mercari_df['cat_so'])
    sparse_matrix_list = (x_name, x_descp, x_brand, x_item_cond_id, x_shipping, x_cat_dae, x_cat_jung, x_cat_so)
    return sparse_matrix_list

def rmsle(y,y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_pred),2)))

def evaluate_org_price(y_test,preds):

    preds_exmpm=np.expm1(preds)
    y_test_exmpm=np.expm1(y_test)

    rmsle_result=rmsle(y_test_exmpm,preds_exmpm)
    return rmsle_result

def model_train_predict(model, matrix_list):
    x=hstack(matrix_list).tocsr()

    x_train,x_test,y_train,y_test=train_test_split(x,mercari_df['price'],test_size=0.2,random_state=36)

    model.fit(x_train,y_train)
    preds=model.predict(x_test)

    del x,x_train,x_test,y_train
    gc.collect()

    return preds,y_test
    


if __name__=='__main__':
    sparse_matrix_list=preprocessing(mercari_df)

    linear_model=Ridge(solver='lsqr',fit_intercept=False)
    lgbm_model=LGBMRegressor(n_estimators=200, learning_rate=0.5, num_leaves=125,random_state=36)

    # sparse_matrix_list = (x_name, x_brand, x_item_cond_id, x_shipping, x_cat_dae, x_cat_jung, x_cat_so)
    # pred,y_test=model_train_predict(linear_model, sparse_matrix_list)
    # print('NO Item Descroption Ridge RMSLE: ', evaluate_org_price(y_test,pred))

    # sparse_matrix_list = (x_name, x_descp, x_brand, x_item_cond_id, x_shipping, x_cat_dae, x_cat_jung, x_cat_so)
    ridge_preds, y_test = model_train_predict(linear_model, sparse_matrix_list)
    print('YES Item Descroption Ridge RMSLE: ', evaluate_org_price(y_test, ridge_preds))

    # sparse_matrix_list = (x_name, x_descp, x_brand, x_item_cond_id, x_shipping, x_cat_dae, x_cat_jung, x_cat_so)
    lgbm_preds, y_test = model_train_predict(lgbm_model, sparse_matrix_list)
    print('YES Item Descroption Ridge RMSLE: ', evaluate_org_price(y_test, lgbm_preds))

    preds=lgbm_preds*0.45+ridge_preds*0.55
    print('Ridge+LGBM ensenble result : ', evaluate_org_price(y_test,preds))