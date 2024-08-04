import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 9999

# read data
path='D:/python/glass_git/Study/Python/Paper/order_batching_optimization/dataset'

dist_matrix=pd.read_csv(path+'/Distance matrix/DistanceMatrix_24SubAisles.txt',delimiter='\t',header=None,engine='python')
dist_matrix=dist_matrix.loc[:,:2880]

#orderlist
orderlist_clolumn=['OrderID', 'NumberOfOrderLines', 'ReleaseTime', 'DueTime' ,'FirstOrderLineID']
orderlist=pd.read_csv(path+'/Orderlist/OrderList_LargeProblems_1_21.txt',delimiter='\t',header=None,engine='python')
orderlist.columns=orderlist_clolumn

#orderline
orderlinelist_column=['OrderID', 'OrderLineID', 'AisleID' ,'CellID', 'LevelID', 'LocationID']
orderlinelist=pd.read_csv(path+'/Orderlinelist/OrderLineList_LargeProblems_1_21.txt',delimiter='\t',header=None,engine='python')
orderlinelist.columns=orderlinelist_column

#---------------------------------------------------------------------------------------------------------------------------------
def distance(aisle_last,max_aisle):
    return (2*50*aisle_last + 500*(max_aisle+1) + (max_aisle-1)*2*50 + 500*max_aisle)/2



if __name__=='__main__':
    orderlist_ix=orderlist.index

    orderlist_ix25=np.random.choice(orderlist_ix,25)
    orderlist_ix=list(set(orderlist_ix)-set(orderlist_ix25))
    print(orderlist.loc[orderlist_ix25]['OrderID'].values)