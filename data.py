import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

#['Unnamed: 0', 'Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA',
# 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
#'Fracture', 'Support Devices']
label_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly','Lung Opacity', 
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia','Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other','Fracture', 'Support Devices']

# create 14 separate dfs, one for each disease

data_path = '/central/groups/CS156b/data/'
train_path = data_path + 'student_labels/train.csv'
test_path = data_path + 'student_labels/test_ids.csv'
imagesize = (50,50)

cuda0 = torch.device('cuda:0')

def train_df():
    return pd.read_csv(train_path)

def get_train_data():
    df = train_df().fillna(0)
    X = []
    Y = []

    for index, row in df.iterrows():
        image_path = data_path + row['Path']

        image = Image.open(image_path).resize(imagesize)
        image_gray = ImageOps.grayscale(image)

        X.append(image_gray)
        Y.append(row[label_columns])
    return X, Y

def trainload(X, Y):
    for i in range(len(X)):
        X[i] = T.ToTensor(X[i])
    trainx = torch.Tensor(X, dtype=torch.float32, device=cuda0)
    trainy = torch.Tensor(Y, dtype=torch.float32, device=cuda0)

    traindata = data_utils.TensorDataset(trainx, trainy)
    train_loader = DataLoader(traindata, batch_size=256, shuffle=True)
    return traindata, train_loader

#def main():
    #X, Y = get_train_data()
    #print(len(X))
    #print(len(Y))

#if __name__ == '__main__':
    
