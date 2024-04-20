from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
import gc
import warnings
warnings.filterwarnings('ignore')

mercari_df=pd.read_csv('train.tsv', sep='\t')
def data_info(mercari_df):
    print('train data shape:  ', mercari_df.shape,end='\n')
    print(tabulate(mercari_df.head(),headers=mercari_df.columns),end='\n\n')
    print('#train data info#: ',end='\n')
    print(mercari_df.info(),end='\n\n')

    print('#Value counts#',end='\n')
    print(mercari_df['item_condition_id'].value_counts(),end='\n\n')
    print(mercari_df['category_name'].value_counts(),end='\n\n')
    print(mercari_df['brand_name'].value_counts(),end='\n\n')
    print(mercari_df['shipping'].value_counts(),end='\n\n')

    no_description=mercari_df['item_description']=='No description yet'
    print('No description: ', mercari_df[no_description]['item_description'].count(),end='\n\n')

    train_y=np.log1p(mercari_df['price'])
    plt.figure(figsize=(6,4))
    sns.distplot(train_y,kde=False)
    plt.show()

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

if __name__=='__main__':
    # data_info(mercari_df)

    mercari_df['cat_dae'],mercari_df['cat_jung'],mercari_df['cat_so']=\
    zip(*mercari_df['category_name'].apply(lambda x: split_cat(x)))
    print('대분류: \n',mercari_df['cat_dae'].value_counts())
    print('중분류: \n', mercari_df['cat_jung'].value_counts())
    print('소분류: \n', mercari_df['cat_so'].value_counts())

    mercari_df['brand_name']=mercari_df['brand_name'].fillna(value='Other_Null')
    mercari_df['category_name'] = mercari_df['category_name'].fillna(value='Other_Null')
    mercari_df['item_description'] = mercari_df['item_description'].fillna(value='Other_Null')
    print(mercari_df.isnull().sum())

    cnt_vec=CountVectorizer()
    x_name=cnt_vec.fit_transform(mercari_df.name)

    tfidf_descp=TfidfVectorizer(max_features=50000,ngram_range=(1,3),stop_words='english')
    x_descp=tfidf_descp.fit_transform(mercari_df['item_description'])

    print('name vect shape: ', x_name.shape)
    print('item_des vect shape: ', x_descp.shape)

    lb_brand_name=LabelBinarizer(sparse_output=True)
    x_brand=lb_brand_name.fit_transform(mercari_df['brand_name'])
    lb_item_cond_id=LabelBinarizer(sparse_output=True)
    x_item_cond_id=lb_item_cond_id.fit_transform(mercari_df['item_condition_id'])
    lb_shipping=LabelBinarizer(sparse_output=True)
    x_shipping=lb_shipping.fit_transform(mercari_df['shipping'])

    lb_cat_dae=LabelBinarizer(sparse_output=True)
    x_cat_dae=lb_cat_dae.fit_transform(mercari_df['cat_dae'])
    lb_cat_jung=LabelBinarizer(sparse_output=True)
    x_cat_jung=lb_cat_jung.fit_transform(mercari_df['cat_jung'])
    lb_cat_so=LabelBinarizer(sparse_output=True)
    x_cat_so=lb_cat_so.fit_transform(mercari_df['cat_so'])

    print(type(x_brand),type(x_item_cond_id),type(x_shipping))
    print('x_brand_shape : ', x_brand.shape, 'x_item_cond_id_shape :' , x_item_cond_id.shape)
    print('x_shipping_shape : ', x_shipping.shape, 'x_cat_dae_shape :', x_cat_dae.shape)
    print('x_cat_jung_shape : ', x_cat_jung.shape, 'x_cat_so_shape :', x_cat_so.shape)

    sparse_matrix_list=(x_name,x_descp,x_brand,x_item_cond_id,x_shipping,x_cat_dae,x_cat_jung,x_cat_so)
    x_feature_sparse=hstack(sparse_matrix_list).\
        tocsr()
    print(type(x_feature_sparse),x_feature_sparse.shape)

    del x_feature_sparse
    gc.collect()