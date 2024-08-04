import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import plotly.express as px
import math
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import Hyperparameter
import itertools 
import matplotlib.cm as cm 
from smt.sampling_methods import LHS
import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax
import time
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", 1000)
init_notebook_mode(connected=True)


class CREATE_DATA:
    def __init__(self,r1,r2,interval=None,num=None):
        self.r1=r1
        self.r2=r2
        self.interval=interval
        self.num=num

    
    def peak_function(self,x1,x2):
        y= np.cos(x1)*np.cos(x2)*(np.exp(-(x1+2)**2-(x2-2)**2)+0.5*np.exp(-(x1-2)**2-(x2+2)**2))
        return y

    def making_Grid(self):
        x1=np.arange(self.r1,self.r2+0.1,self.interval)
        x2=np.arange(self.r1,self.r2+0.1,self.interval)
        data=pd.DataFrame()
        for i in x1 :
            for z in x2:
                a=[[i,z]]
                data=data.append(a,ignore_index=True)
        data.columns=['x1','x2']
        data['y']=self.peak_function(data['x1'],data['x2'])

        return data

    def making_Random(self,r1,r2,num):
        x={'x1':np.random.uniform(self.r1,self.r2,self.num).T, 'x2':np.random.uniform(self.r1,self.r2,self.num).T}
        data=pd.DataFrame(x)
        data['y']=self.peak_function(data['x1'],data['x2'])
        return data

    def making_LHS(self,r1,r2,num):
        xlimits = np.array([[self.r1, self.r2], [self.r1, self.r2]])
        sampling = LHS(xlimits=xlimits,criterion='ese')
        x = sampling(self.num)
        data=pd.DataFrame(x,columns=['x1','x2'])
        data['y']=self.peak_function(data['x1'],data['x2'])
        return data


class QBC:
    def __init__(self,Committee_num):
        self.committee_num=Committee_num

    def peak_function(self,x1,x2):
        y= np.cos(x1)*np.cos(x2)*(np.exp(-(x1+2)**2-(x2-2)**2)+0.5*np.exp(-(x1-2)**2-(x2+2)**2))
        return y

    def committee(self,data,a):
        qbc1=data.sample(n=len(data),replace=True)

        x_qbc1=qbc1[['x1','x2']]
        y_qbc1=qbc1['y']

        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RationalQuadratic(length_scale=1.0, alpha=1.5)
        qbc_model1 = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=a, normalize_y=True,random_state=42)
        qbc_model1.fit(x_qbc1, y_qbc1)
        return qbc_model1

    def LOO(self,lhs_init):
        MSE=pd.DataFrame()
        for i in range(len(lhs_init)):
            x_true=lhs_init[['x1','x2']]
            y_true=lhs_init['y'] 

            sub_x=x_true.drop(x_true.index[i:i+1])
            sub_y=y_true.drop(y_true.index[i:i+1])

            #sub를 제외한 나머지 labeled data로 모델 생성
            r2,rmse=self.model_gp(sub_x,sub_y,x_true,y_true)
            MSE=MSE.append([rmse],ignore_index=True)

        MSE.columns=['rmse']
        var_df=pd.merge(lhs_init,MSE,left_index=True,right_index=True)
        var_df['probability']=var_df['rmse']/sum(var_df['rmse'])
        var_df.sort_values(by=['rmse'],ascending=False,inplace=True)
        var_df.reset_index(drop=True,inplace=True)
        return var_df

    def model_gp(self,train_x,train_y,test_x,test_y):
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RationalQuadratic(length_scale=1.0, alpha=1.5)
        model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True,random_state=42)
        model.fit(train_x, train_y)
        y_pred, std = model.predict(test_x, return_std=True)   
        rmse=mean_squared_error(test_y,y_pred,squared=False)
        r2=r2_score(test_y,y_pred)
        
        return r2,rmse


    def training(self,candidates_num,newpoints_file,result_file,prob,save_train):
        test_df=pd.read_csv('D:\\Active-learning\\peak1_test10000.csv')
        test_x=test_df.drop('y',axis=1)
        test_y=test_df['y']
        loss = pd.DataFrame()
        for z in range(1):
            lhs_init=pd.read_csv('D:\\Active-learning\\new_lhs_init_peak1.csv')
            NEW_POINTS=pd.DataFrame()

            init_n=0
            start=datetime.now()
            while len(lhs_init)<81:

                #test set에서 랜덤으로 20개 가져오기
                candidates=test_df.sample(20)
                candidates.reset_index(drop=True, inplace=True)
                #y버리기
                candi_x=candidates.drop('y',axis=1)

                #qbc 모델 만들기
                model_params=[0.1,0.2,0.3,0.4,0.5]
                for i in range(self.committee_num):
                    globals()['qbc{}'.format(i+1)]=self.committee(lhs_init,model_params[i])

                #후보군 중에서 1개 선택

                var=[]
                for i in range(len(candi_x)):
                    x_loo=candi_x[i:i+1]

                    p=[]
                    for i in range(self.committee_num):
                        pred=globals()['qbc{}'.format(i+1)].predict(x_loo)
                        p.append(pred)

                    v=np.var(p)
                    var.append(v)
                Var=pd.DataFrame(data=var,columns=['var'])
                can_prior=pd.merge(candidates,Var,left_index=True,right_index=True)

                can_prior['probability']=can_prior['var']/sum(can_prior['var'])
                can_prior['p_root']=(can_prior['var']/sum(can_prior['var']))**0.5
                can_prior['p_square']=(can_prior['var']/sum(can_prior['var']))**2
                init_points=lhs_init[['x1','x2']]
                dist=pd.DataFrame(distance.cdist(init_points, candi_x, 'euclidean'))

                # 이니셜 과 새로운 후보들 간의 평균 거리 중 가장 멀리 있는 후보 선택
                dist_mean=pd.DataFrame(dist.mean(),columns=['dist_mean'])

                # 후보군과 가장 가까운 이니셜과의 거리가 가장 먼것 
                dist_min=pd.DataFrame(dist.min(),columns=['dist_min'])

                can_prior=pd.merge(can_prior,dist_mean,right_index=True, left_index=True)
                can_prior=pd.merge(can_prior,dist_min,right_index=True, left_index=True)
                can_prior['dist_mean_prob'] = can_prior['dist_mean'] / sum(can_prior['dist_mean'])
                can_prior['dist_min_prob'] = can_prior['dist_min'] / sum(can_prior['dist_min'])

                
                can_prior['v*mean*min']=(can_prior['probability'])*(can_prior['dist_mean_prob'])*(can_prior['dist_min_prob'])
                #can_prior['v*mean*min11']=(can_prior['probability']*0.3)+(can_prior['dist_mean_prob']*0.3)+(can_prior['dist_min_prob']*0.4)
                #can_prior['v*mean*min12']=(can_prior['probability']*(0.1+init_n*(0.3/(81-20))))*(can_prior['dist_mean_prob']*(0.3-init_n*(0.1/(81-20))))*(can_prior['dist_min_prob']*(0.6-init_n*(0.2/(81-20))))
                
                can_prior['vmm_prob']=can_prior[prob] / sum(can_prior[prob])

                can_prior.sort_values(by=['vmm_prob'],ascending=False,inplace=True)
                can_prior.reset_index(drop=True,inplace=True)

                new_x=can_prior.iloc[0]
                new_x['y']=self.peak_function(new_x['x1'],new_x['x2'])
                new_point=new_x[['x1','x2','y']]
                new_point.columns=['x1','x2','y']

                lhs_init=lhs_init.append(new_point,ignore_index=True)
                NEW_POINTS=NEW_POINTS.append(new_point,ignore_index=True)  

                #r2확인하기
                train_x=lhs_init.drop('y',axis=1)
                train_y=lhs_init['y']

                r2,rmse=self.model_gp(train_x,train_y,test_x,test_y)
                init_n+=1
                print('points:',len(lhs_init),'   r2:',r2, '    rmse: ',rmse) 
                    
            finish=datetime.now()
            run_time=finish-start
            print(run_time)
            print('total points:',len(lhs_init),'   r2:',r2, '    rmse: ',rmse) 
                
            loss=loss.append([[r2,rmse]],ignore_index=True)

            # 결과 파일
            self.save_newpoints(newpoints_file,z,NEW_POINTS)  
            self.save_newpoints(save_train,z,lhs_init) 

            #그래프
            self.graph(NEW_POINTS)


        print(prob, loss.mean())

        self.save_result(result_file,candidates_num,prob,run_time,loss) 
            
        return loss


    #새로운 data point 저장
    def save_newpoints(self,file_name,z,NEW_POINTS):
        f = open(file_name,'a')
        time = "%d TIME.\n\n" %z
        f.write(time)
        f.close()
        result=pd.read_csv(file_name, sep = '\t',encoding='EUC-KR')
        result=result.append(NEW_POINTS)
        result.to_csv(file_name, sep = '\t',index=False)
        return 

    #결과 파일 저장
    def save_result(self,file_name,candidates_num,prob,run_time,loss):
        f = open(file_name,'a')
        time = "candidates_num : %d,     prob : %s,  run_time : %s \n\n" %(candidates_num,prob,run_time)
        f.write(time)
        f.close()
        result=pd.read_csv(file_name, sep = '\t',encoding='EUC-KR')
        result=result.append(loss)
        result.to_csv(file_name, sep = '\t',index=True)
        return 

    #그래프
    def graph(self,NEW_POINTS):
        init=pd.read_csv('new_lhs_init_peak1.csv')

        X=NEW_POINTS['x1']
        Y=NEW_POINTS['x2']
        labels=range(0,len(NEW_POINTS))

        fig = plt.figure()
        ax = fig.add_subplot(111)


        for x,y,lab in zip(X,Y,labels):
                ax.scatter(x,y,label=lab)

        colormap = plt.cm.Reds
        colorst = [colormap(i) for i in np.linspace(0.9, 0,len(ax.collections))]       
        for t,j1 in enumerate(ax.collections):
            j1.set_color(colorst[t])

        return ax.scatter(init['x1'], init['x2'], s=10, c='b', marker="s", label='init')
    

    if __name__=="__main__":
        candidates_num=20
        newpoints_file='D:\\Active-learning\\qbc_vmm\\qbc_newpoint.txt'
        result_file='D:\\Active-learning\\qbc_vmm\\qbc_loss.txt'
        prob='probability'
        save_train='D:\\Active-learning\\qbc_vmm\\qbc_train_all81.txt'

        QBC.training(candidates_num,newpoints_file,result_file,prob,save_train)

