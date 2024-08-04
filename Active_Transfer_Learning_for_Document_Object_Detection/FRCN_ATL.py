import torch
import torchvision 
import pandas as pd
import numpy as np
import argparse
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
import utils_ObjectDetection as utils
from torch.utils.data import BatchSampler,SequentialSampler,RandomSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import math 
from itertools import combinations
import ITB_RECALL_PRECISION_F1 as rp_f1


   
class simpleDataset(object):


    def __init__(self, dataset,resize,color, img_foler_path,transforms=None):
    #             self.root = root
            self.transforms = transforms
            self.adnoc = dataset
            self.ids = dataset.index
            self. filenames=dataset['path'].to_list()
            self.resize =resize
            self.color=color
            self.img_foler_path=img_foler_path
            
    def __getitem__(self, index):
        # Own coco file
        adnoc_df = self.adnoc
        # Image ID ,폴더에서 index번째 
        img_id = self.ids[index]

       # List: get annotation id from coco, index번째 annotation가져오기 [[1,2,3,4],[5,6,7,8]]
        annotation = adnoc_df['box'][img_id]


        # open the input image, 흑백=L , 단색=1
        img= Image.open(str(self.img_foler_path)+str(adnoc_df.loc[img_id]['path'])).convert(self.color)
        img= img.resize((int(img.width / self.resize), int(img.height / self.resize)))
      
        # number of objects in the image
        num_objs = len(annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        areas = []
        label = []
        for i in range(num_objs):

            xmin = annotation[i][0]
            ymin = annotation[i][1]
            xmax = xmin + annotation[i][2]
            ymax = ymin + annotation[i][3]
            l=annotation[i][4]
            area=annotation[i][2]*annotation[i][3]

            boxes.append([xmin/self.resize, ymin/self.resize, xmax/self.resize, ymax/self.resize])
          
            label.append(l)
            areas.append(area)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #areas= box size = width * height
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(label, dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)



    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)

    
class ATL_FRCN:

    def __init__(self,train,valid,test,total,method,method_param,param_name,TL_param,Train_param_path,result_path,sample_path):
        self.train=train
        self.valid=valid
        self.test=test
        self.total=total
        self.method=method
        self.method_param=method_param
        self.param_name=param_name
        self.TL_param=TL_param
        self.Train_param_path=Train_param_path
        self.result_path=result_path
        self.sample_path=sample_path


    def make_prediction(self,model, img, threshold):
        model.eval()
        preds = model(img)
        for id in range(len(preds)) :
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']) :
                
                if score > threshold : 
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list]
            preds[id]['labels'] = preds[id]['labels'][idx_list]
            preds[id]['scores'] = preds[id]['scores'][idx_list]

        return preds

    def collate_fn(self,batch):
        return tuple(zip(*batch))


    def get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)


    def get_model_instance_segmentation(self,num_classes,train_layer):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=train_layer)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def plot_image_from_output(self,img, annotation):
        
        img = img.cpu().permute(1,2,0)
        
        fig,ax = plt.subplots(figsize=(20, 10))
        ax.imshow(img)
        
        for idx in range(len(annotation["boxes"])):
            xmin, ymin, xmax, ymax = annotation["boxes"][idx]

            if annotation['labels'][idx] == 1 :
                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')
            
            elif annotation['labels'][idx] == 2 :
                
                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
                
            elif annotation['labels'][idx] == 3 :
            
                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')
            else:
                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

            ax.add_patch(rect)

        return plt


 
    def data_load(self,data_file_path,file_name):
        """
        data_file_path : 'D:/OBJ_PAPER/'
        file_name : 'itb.pkl'
        """
        file_path=data_file_path 
        itb=pd.read_pickle(file_path+file_name)

        test=itb.sample(frac=0.2,random_state=42)
        train_valid=itb.drop(test.index)
        valid=train_valid.sample(n=50, random_state=42)
        train=train_valid.drop(valid.index)


        test.reset_index(drop=True,inplace=True)
        test['index']=test.index

        valid.reset_index(drop=True,inplace=True)
        valid['index']=valid.index

        train.reset_index(drop=True,inplace=True)
        train['index']=train.index
        print(' train shape : ',train.shape, 'valid shape : ', valid.shape, 'test shape : ', test.shape, )

        # create own Dataset
        train_dataset = simpleDataset(dataset=train,
                                    resize=4,
                                    color='L',
                                    img_foler_path=file_path+'Data/itb_p_500/',
                                    transforms=self.get_transform())

        valid_dataset = simpleDataset(
                                dataset=valid,
                                resize=4,
                                color='L',
                                img_foler_path=file_path+'Data/itb_p_500/',
                                transforms=self.get_transform())

        test_dataset = simpleDataset(
                                dataset=test,
                                resize=4,
                                color='L',
                                img_foler_path=file_path+'Data/itb_p_500/',
                                transforms=self.get_transform())



        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,                                         
                                                collate_fn=self.collate_fn)

        valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=0,
                                                collate_fn=self.collate_fn)

        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=self.collate_fn)

        return test,valid,train,train_data_loader,valid_data_loader, test_data_loader


    def train_AL(self,num_classes,num_epochs,train_layer,batch,patience,ascend):  
        
      
        model = self.get_model_instance_segmentation(num_classes,train_layer)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # move model to the right device
        model.to(device)
        model.load_state_dict(torch.load(self.TL_param))

        # parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.0001)
        
        iteration = 0    

        RESULT=pd.DataFrame()
        Sampled_img=pd.DataFrame()
        
        #모델 저장 
        self.train_param_path=self.Train_param_path.format(self.total,self.method,self.param_name)
        
        #data reload
    
        train_data_loader=self.data_load(self.train)
        valid_data_loader=self.data_load(self.valid)
        test_data_loader=self.data_load(self.test)
        
        while len(Sampled_img)<self.total:

            if len(Sampled_img)+batch>self.total:
                batch=(len(Sampled_img)+batch)-self.total
                sample=self.train.sample(n=batch)
                Sampled_img=Sampled_img.append(sample,ignore_index=True)
                sample_data_loader=self.data_load(Sampled_img)
                
            else:
                us=self.uncertainty_sampling(model,0.00001,batch,train_data_loader,self.method,self.method_param)

                model_pred,sampled,train_rest,train_rest_data_loader=us.sampling(self.train,ascend)

                Sampled_img=Sampled_img.append(sampled,ignore_index=True)  
                sample_data_loader=self.data_load(Sampled_img)
                
                #sample 제외한 나머지 train 데이터
                train=train_rest

                #ranodm으로 가져온 N개 데이터
                train_data_loader=train_rest_data_loader


            iteration += 1
            print(iteration)
            model.train()
            not_save_count=0
            
            #평균 loss
            avg_train_loss=[]
            avg_valid_loss=[]
            
            for epoch in range(num_epochs):

                # 모델이 학습되는 동안 trainning loss를 track
                train_losses = []
                # 모델이 학습되는 동안 validation loss를 track
                valid_losses = []
                
                st = time.time()
                for imgs, annotations in sample_data_loader:

                    imgs = list(img.to(device) for img in imgs)
                    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                    loss_dict = model(imgs, annotations)
                    #batch size=10, 10개 loss 각각 도출
                    losses = sum(loss for loss in loss_dict.values())
                    train_losses.append(losses.item())       
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                #이미지 한장당 평균 loss
                avg_train_loss.append(np.mean(train_losses).round(5))

                #validation, early_stop, save weights
                with torch.no_grad():

                    for im, annot in valid_data_loader:
                        im = list(img.to(device) for img in im)
                        annot = [{k: v.to(device) for k, v in t.items()} for t in annot]
                        val_loss_dict = model(im, annot)
                        val_losses = sum(val_loss for val_loss in val_loss_dict.values())
                        valid_losses.append(val_losses.item())

                    epoch_val_loss=np.mean(valid_losses).round(5)
                    avg_valid_loss.append(epoch_val_loss)  

                fi = time.time()     
                print('epoch:',epoch,'train_loss: ',np.mean(train_losses).round(5),'validation loss : ',np.mean(valid_losses).round(5),'time',fi-st)


                min_val_loss=np.min(avg_valid_loss)
                if min_val_loss>=epoch_val_loss:

                    torch.save(model.state_dict(),self.train_param_path)
                    not_save_count=0
                    print('epoch:',epoch,'save model')

                else:
                    not_save_count+=1
                    model.load_state_dict(torch.load(self.train_param_path))
                    if not_save_count>=patience:
                        print('no more training')
                        break


            fi = time.time()     
            print('iteration:',iteration,'train_loss: ',np.mean(train_losses).round(5),'time',fi-st)
            print('sample num:',len(Sampled_img))
            model.load_state_dict(torch.load(self.train_param_path))

            result_batch=self.result_RPA(test_data_loader,model,self.train_param_path,iteration)
            RESULT=RESULT.append(result_batch,ignore_index=True)

        RESULT.to_csv(self.result_path.format(self.total,self.method,self.param_name),index=False)   
        Sampled_img.to_csv(self.sample_path.format(self.total,self.method,self.param_name),index=False) 
        return RESULT
    
    if __name__=="__main__":

        #read data sets
        data_path='D:/OBJ_PAPER/Data/3_fold_cv/'

        train1=pd.read_pickle(data_path+'train1.pkl')
        train2=pd.read_pickle(data_path+'train2.pkl')
        train3=pd.read_pickle(data_path+'train3.pkl')

        valid1=pd.read_pickle(data_path+'valid1.pkl')
        valid2=pd.read_pickle(data_path+'valid2.pkl')
        valid3=pd.read_pickle(data_path+'valid3.pkl')

        test=pd.read_pickle(data_path+'test.pkl')
                

        TL_param='D:/OBJ_PAPER/model_param/model_fasterrcnn_bach5_p10_50_param.pt'
        Train_param_path='D:/OBJ_PAPER/model_param/FRCN_AL1_batch30_total{}_{}_{}param.pt'
        result_path='D:/OBJ_PAPER/result/FRCN_AL1_batch30_total{}_{}_{}_result.csv'
        sample_path='D:/OBJ_PAPER/result/FRCN_AL1_batch30_total{}_{}_{}_sampled.csv'

        train=train1
        valid=valid1
        test=test

        total=180
        Method= [
            'class_difficulty',
            'class_ambiguity',
        ]

        a=[0.7,0.9]
        b=[0.1,0.5]
        c=[0.7,0.9]

        num_classes = 4
        num_epochs = 100
        train_layer=5
        batch=30
        patience=10
        ascend=False
        #param=[un_param,diff_param,ambi_param]

        for method in Method:
            if method=='uncertainty':
                for a_param in a:
                    print('total:',total,'method:',method,'a_param:',a_param)
                    method_param=[a_param,0.3,0.5]
                    atl=ATL_FRCN(train,valid,test,total,method,method_param,a_param,TL_param,Train_param_path,result_path,sample_path)
                    atl.train_AL(num_classes,num_epochs,train_layer,batch,patience,ascend)
                    
            elif method=='class_difficulty':
                for b_param in b:
                    print('total:',total,'method:',method,'b_param:',b_param)
                    method_param=[0.5,b_param,0.5]
                    atl=ATL_FRCN(train,valid,test,total,method,method_param,a_param,TL_param,Train_param_path,result_path,sample_path)
                    atl.train_AL(num_classes,num_epochs,train_layer,batch,patience,ascend)

            elif method=='class_ambiguity':
                for c_param in c:
                    print('total:',total,'method:',method,'c_param:',c_param)
                    atl=ATL_FRCN(train,valid,test,total,method,method_param,a_param,TL_param,Train_param_path,result_path,sample_path)
                    atl.train_AL(num_classes,num_epochs,train_layer,batch,patience,ascend)
        
        
        
