import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow_datasets as tfds
ds = tfds.load('the300w_lp', split='train')
ds=ds.batch(64)

import torch
import torch.nn as nn
from PIL import Image
from torchsummary import summary
import torchvision.models as models
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from math import*

torch.cuda.empty_cache()
device=torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class HeadPoseEstimation(nn.Module):
    def __init__(self):
        super(HeadPoseEstimation, self).__init__()
        efficientnet = models.efficientnet_b0()
        efficientnet=torch.nn.Sequential(*(list(efficientnet.children())[:-1]),
                                        nn.Flatten())
        self.efficientnet = efficientnet
        self.LinearPitch66=nn.Linear(1280,66)
        self.LinearRoll66=nn.Linear(1280,66)
        self.LinearYaw66=nn.Linear(1280,66)
        self.Softmax=nn.Softmax(dim=1)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.efficientnet(x)
        
        yaw=self.LinearYaw66(x)
        yaw=self.Softmax(yaw)
        
        pitch=self.LinearPitch66(x)
        pitch=self.Softmax(pitch)
        
        roll=self.LinearRoll66(x)
        roll=self.Softmax(roll)
        return yaw,pitch,roll
    
model=HeadPoseEstimation()
try:
    model=torch.load('WHENet.pt')
except:
    pass
model=model.to(device)

MSEloss = nn.MSELoss()
CEloss = nn.CrossEntropyLoss()
def MAELoss(y_pred,y_true):
    return abs(y_pred-y_true).mean()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

for param in model.parameters():
    param.requires_grad=True
  
ClassificationTrainLoss=[]
ClassificationValidLoss=[]
RegressionTrainLoss=[]
RegressionValidLoss=[]
MAELossPitch=[]
MAELossYaw=[]
MAELossRoll=[]

num_epochs=10
num_batch=900
for epoch in range(num_epochs):
    it=ds.as_numpy_iterator()
    
    for i, x in enumerate(ds):
        x=it.next()
        
        # Forward pass
        batch_input=torch.tensor(np.array([resize(img,(224,224)).reshape(3,224,224) for img in x['image']]),dtype=torch.float ).to(device)
        pred_yaw, pred_pitch, pred_roll=model(batch_input)
        
        batch_output_pitch,batch_output_yaw,batch_output_roll=torch.tensor(((x['pose_params'][:,:3]*180/np.pi).T+99)/3,dtype=torch.long).to(device)

        # Compute Loss
        Loss_classification = (CEloss(pred_pitch,batch_output_pitch) + CEloss(pred_yaw, batch_output_yaw)+CEloss(pred_roll, batch_output_roll))/3
        
        pred_pitch_regression=(torch.tensor([range(pred_pitch.shape[1]) for _ in range(pred_pitch.shape[0])]).to(device)*pred_pitch).sum(axis=1)
        pred_yaw_regression=(torch.tensor([range(pred_yaw.shape[1]) for _ in range(pred_yaw.shape[0])]).to(device)*pred_yaw).sum(axis=1)
        pred_roll_regression=(torch.tensor([range(pred_roll.shape[1]) for _ in range(pred_roll.shape[0])]).to(device)*pred_roll).sum(axis=1)
        Loss_regression=(MSEloss(pred_pitch_regression,batch_output_pitch.float())+2*MSEloss(pred_yaw_regression,batch_output_yaw.float())+MSEloss(pred_roll_regression,batch_output_roll.float()))/3
        
        loss=2*Loss_classification+2*Loss_regression
        
        if i>num_batch:
            ClassificationValidLoss.append(Loss_classification.item())
            RegressionValidLoss.append(Loss_regression.item())
            
            MAELossPitch.append(MAELoss(pred_pitch_regression,batch_output_pitch.float()).item())
            MAELossYaw.append(MAELoss(pred_yaw_regression,batch_output_yaw.float()).item())
            MAELossRoll.append(MAELoss(pred_roll_regression,batch_output_roll.float()).item())
            optimizer.zero_grad()
            print('Valid process saved\n')
            break

        # Backward and optimizes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_batch}], Loss: {loss.item():.4f}, Loss_cls: {Loss_classification.item():.4f}, Loss_regression: {Loss_regression.item():.4f}')
            if i<10:
                ClassificationTrainLoss.append(Loss_classification.item())
                RegressionTrainLoss.append(Loss_regression.item())
                
                print('Train process saved\n')

    pkl.dump((ClassificationTrainLoss, ClassificationValidLoss, RegressionTrainLoss, RegressionValidLoss,MAELossPitch,MAELossYaw,MAELossRoll),open('process.pkl','wb'))
    torch.save(model, 'WHENet.pt')