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
imagesize = (64,64)

cuda0 = torch.device('cuda:0')

def get_train_data():
    df = pd.read_csv(train_path).fillna(0)
    X = []
    Y = []

    for index, row in df.iterrows():
        if index == int(df.shape[0]/100):
            break
        image_path = data_path + row['Path']

        image = Image.open(image_path)
        imcopy = image.copy()
        #.resize(imagesize)
        image_gray = ImageOps.grayscale(imcopy)

        X.append(imcopy)
        image.close()
        Y.append(row[label_columns])
    return X, Y

def get_test_data():
    df = pd.read_csv(test_path)
    X = []
    for index, row in df.iterrows():
        image_path = data_path + row['Path']

        image = Image.open(image_path)
        #.resize(imagesize)
        #image_gray = ImageOps.grayscale(image)

        X.append(image)
    trans = T.Compose([T.ToTensor(), T.Resize(size=(64,64))])
    for i in range(len(X)):
       X[i] = trans(X[i]).numpy()
    testx = torch.from_numpy(np.array(X).astype(np.float32))
    testx.to(cuda0)

    testloader = DataLoader(testx, batch_size=1, shuffle=False)
    return testloader

def trainload(X, Y):
    trans = T.Compose([T.ToTensor(), T.Resize(size=(64,64))])
    for i in range(len(X)):
       X[i] = trans(X[i]).numpy()
    #trainx = torch.Tensor(X).cuda(device=cuda0)
    #trainy = torch.Tensor(Y).cuda(device=cuda0)
    trainx = torch.from_numpy(np.array(X).astype(np.float32))
    trainy = torch.from_numpy(np.array(Y).astype(np.float32))
    trainx.to(cuda0)
    trainy.to(cuda0)

    traindata = data_utils.TensorDataset(trainx, trainy)
    train_loader = DataLoader(traindata, batch_size=256, shuffle=True)
    return traindata, train_loader

#def main():
    #X, Y = get_train_data()
    #print(len(X))
    #print(len(Y))

#if __name__ == '__main__':
    
