from pycm import *
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import pickle
import sys
from glob import glob  
import math
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
import torch.utils.data.dataloader
import torchvision.transforms as visionTransforms
import PIL.Image as Image
from torchvision.transforms import ToTensor,ToPILImage
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import io
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dfTrain = pd.read_csv("/root/combined.csv", index_col=None)
dfTest=pd.read_csv("/root/test.csv",escapechar = "\\",quoting = csv.QUOTE_NONE)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

labelEncoder=preprocessing.LabelEncoder()
encodedLabelListTrain=(labelEncoder.fit_transform(dfTrain["Label"]))
dfTrain["Label"]=encodedLabelListTrain

from torch.utils.data import WeightedRandomSampler
freqLabels=torch.tensor(dfTrain['Label'].value_counts().sort_index(),dtype=torch.double)
weightClass=freqLabels/freqLabels.sum()
weightClass= 1/weightClass
weightClass=(weightClass).tolist()
sampleWeights=[weightClass[i] for i in dfTrain['Label']]
trainSampler=WeightedRandomSampler(sampleWeights,len(dfTrain))

from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
from torch.utils.data import Dataset, DataLoader

class QuoraDataset(Dataset):

  def __init__(self,dataframe,bertTokenizer,maxLength,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.maxLength=maxLength
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    #print(idx)
    self.productDescription=str(self.data.loc[idx,"Text"])
    self.label=self.data.loc[idx,"Label"]

    self.encodedInput=self.bertTokenizer.encode_plus(text=self.productDescription,padding='max_length',truncation="longest_first",max_length=self.maxLength,return_tensors='pt',return_attention_mask=True,return_token_type_ids=True).to(device)
    
    return self.encodedInput,self.label

class FlipkartTestDataset(Dataset):

  def __init__(self,dataframe,bertTokenizer,maxLength,device):
    self.data=dataframe
    self.bertTokenizer=bertTokenizer
    self.maxLength=maxLength
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    self.productDescription=str(self.data.iloc[idx,1])+str(self.data.iloc[idx,2])+str(self.data.iloc[idx,3])

    self.encodedInput=self.bertTokenizer.encode_plus(text=self.productDescription,padding='max_length',truncation="longest_first",max_length=self.maxLength,return_tensors='pt',return_attention_mask=True,return_token_type_ids=True).to(device)
    
    return self.encodedInput

from transformers import MobileBertTokenizer, MobileBertModel

tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
model = MobileBertModel.from_pretrained('google/mobilebert-uncased')

quoraTrainDataset=QuoraDataset(dataframe=dfTrain,bertTokenizer=tokenizer,maxLength=128,device=device)
flipkartTestDataset=FlipkartTestDataset(dataframe=dfTest,bertTokenizer=tokenizer,maxLength=128,device=device)
trainLoader=torch.utils.data.DataLoader(quoraTrainDataset,batch_size=8,sampler=trainSampler)
testLoader=torch.utils.data.DataLoader(flipkartTestDataset,batch_size=8,shuffle=False)

class BERTOnly(nn.Module):
  def __init__(self,preTrainedBert,embeddingDimension=512,numClasses=1):
    super(BERTOnly,self).__init__()

    self.embDim=embeddingDimension
    self.numClasses=numClasses

    self.dropoutLayer=nn.Dropout(p=0.5)
    self.bert=self.freezeBert(preTrainedBert)
    self.fc1=nn.Linear(self.embDim,9919)

  def forward(self,input):
    bertOutput=self.bert(input_ids=input['input_ids'].squeeze(dim=1),attention_mask=input['attention_mask'].squeeze(dim=1)).pooler_output
    #print(bertOutput.shape)
    classificationOutput=self.fc1(self.dropoutLayer(bertOutput))
    #print(classificationOutput.shape)
    #classificationOutput=classificationOutput.reshape((classificationOutput.size(0)))
    #print(classificationOutput.shape)
    return classificationOutput

  def freezeBert(self,model):
    return model

bertOnly=BERTOnly(preTrainedBert=model)
bertOnly.to(device)
softmaxLoss = nn.CrossEntropyLoss()
optimizer = optim.Adam(bertOnly.parameters(), lr=0.00001)

def Average(lst): 
    return sum(lst) / len(lst) 

def train_model(model,epochs):

  trainBatchCount=0
  testBatchCount=0

  avgTrainAcc=[]
  avgValidAcc=[]
  trainAcc=[]
  validAcc=[]
  trainLosses=[]
  validLosses=[]
  avgTrainLoss=[]
  avgValidLoss=[]


  for i in range(epochs):

    print("Epoch:",i)

    model.train()
    print("Training.....")
    for batch_idx,(data,targets) in enumerate(trainLoader):

      trainBatchCount=trainBatchCount+1

      targets=targets.to(device)

      optimizer.zero_grad()

      scores=model(data)
       
      loss=softmaxLoss(scores,targets)

      loss.backward()

      optimizer.step()

      trainLosses.append(float(loss))

      
      correct=0
      total=0
      total=len(targets)


      predictions=torch.argmax(scores,dim=1)
      correct = (predictions==targets).sum()
      acc=float((correct/float(total))*100)

      trainAcc.append(acc)

      if ((trainBatchCount%200)==0):

        print("Targets:-",targets)
        print("Predictions:-",predictions)

        print("Epoch:",i)
        print("Batch:",batch_idx)
        print('Loss: {}  Accuracy: {} %'.format(loss.data, acc))
	

    #model.eval()
    #print("Validating.....")
    #for data,targets in valLoader:

      #testBatchCount=testBatchCount+1

      #targets=targets.to(device=device)

      #scores=model(data)

      #loss=softmaxLoss(scores,targets)

      #validLosses.append(float(loss))

      #testCorrect=0
      #testTotal=0

      #_,predictions=scores.max(1)

      #testCorrect = (predictions==targets).sum()
      #testTotal=predictions.size(0)

      #testAcc=float((testCorrect/float(testTotal))*100)

      #validAcc.append(testAcc)

      #if ((testBatchCount%200)==0):

        #print('Loss: {}  Accuracy: {} %'.format(float(loss), testAcc))
    

    trainLoss=Average(trainLosses)
    #validLoss=Average(validLosses)
    avgTrainLoss.append(trainLoss)
    #avgValidLoss.append(validLoss)
    tempTrainAcc=Average(trainAcc)
    #tempTestAcc=Average(validAcc)
    avgTrainAcc.append(tempTrainAcc)
    #avgValidAcc.append(tempTestAcc)

    print("Epoch Number:-",i,"  ","Training Loss:-"," ",trainLoss,"Training Acc:-"," ",tempTrainAcc)

    trainAcc=[]
    ValidAcc=[]
    trainLosses=[]
    validLosses=[]

  return model,avgTrainLoss,avgTrainAcc

bertOnly,avgTrainLoss,avgTrainAcc = train_model(bertOnly,3)


def checkClassificationMetrics(loader,model):

  completeTargets=[]
  completePreds=[]

  correct=0
  total=0
  model.eval()

  with torch.no_grad():
    for data in loader:

      #targets=targets.to(device=device)

      scores=model(data)
      _,predictions=scores.max(1)

      #targets=targets.tolist()
      predictions=predictions.tolist()

      #completeTargets.append(targets)
      completePreds.append(predictions)

    #completeTargetsFlattened=[item for sublist in completeTargets for item in sublist]
    completePredsFlattened=[item for sublist in completePreds for item in sublist]

    #cm = ConfusionMatrix(actual_vector=completeTargetsFlattened, predict_vector=completePredsFlattened)
    return completePredsFlattened

CM=checkClassificationMetrics(testLoader,bertOnly)

finalResult=labelEncoder.inverse_transform(CM)
pid=list(dfTest.iloc[:,0])

tempdf=pd.DataFrame(list(zip(pid, finalResult)),
               columns =['PRODUCT_ID', 'BROWSE_NODE_ID'])

tempdf.to_csv("Results_MobileBERT.csv")


