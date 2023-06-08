import pandas as pd
import glob, os
from tabulate import tabulate
path=os.getcwd()+'/OpinosisDataset1.0/topics'
all_files=glob.glob(os.path.join(path,'*.data'))
filename_list=[]
opinion_text=[]


for file_ in all_files:
    df=pd.read_table(file_,header=None,index_col=False,encoding='latin1')
    text=''.join([df[0][d] for d in range(len(df[0]))])
    filename_=file_.split('/')[-1]
    filename=filename_.split('.')[0]
    filename_list.append(filename)
    opinion_text.append(text)

document_df=pd.DataFrame({'filename':filename_list,'opinion_text':opinion_text})
print(tabulate(document_df.head(),headers=document_df.columns))