from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

mercari_df=pd.read_csv('train.tsv', sep='\t')
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