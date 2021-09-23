import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from utils import *
from usad import *
from sklearn import preprocessing
from pathlib import Parh

device = get_dafault_device()

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

min_max_scaler = preprocessing.MinMaxScaler()

# TRAIN
train_dataset = sorted([x for x in Path('input/training/').glob("*.csv")])
train = dataframe_from_csvs(train_dataset)
train = train.drop(["time"] , axis = 1)
print("test shape :\t", train.shape)

for i in list(train):
    train[i]=train[i].apply(lambda x: str(x).replace("," , "."))
train = train.astype(float)
train.shape

x = train.values
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)

#TEST
test_dataset = sorted([x for x in Path('input/testing/').glob("*.csv")])
test = dataframe_from_csvs(test_dataset)
test = test.drop(["time"], axis=1)
print("test shape :\t", test.shape)

for i in list(test):
    test[i]=test[i].apply(lambda x: str(x).replace(",", "."))
test = test.astype(float)

x = test.values
x_scaled = min_max_scaler.transform(x)
pd.set_option('display.max_columns', 0)
test = pd.DataFrame(x_scaled)

#VALIDATION
validation_dataset = sorted([x for x in Path('input/validation/').glob("*.csv")])
validation = dataframe_from_csvs(validation_dataset)
labels = [ float(label!= 0 ) for label  in attack["attack"].values]
validation = validation.drop(["time", "attack"] , axis = 1)
print("validation shape :\t", validation.shape)

for i in list(validation):
    validation[i]=validation[i].apply(lambda x: str(x).replace("," , "."))
validation = validation.astype(float)


x = validation.values
x_scaled = min_max_scaler.transform(x)
validation = pd.DataFrame(x_scaled)


# WINDOW_SIZE must be lower than 8
WINDOW_SIZE=7

windows_train=train.values[np.arange(WINDOW_SIZE)[None, :] + np.arange(train.shape[0]-WINDOW_SIZE)[:, None]]
windows_train.shape

windows_test=test.values[np.arange(WINDOW_SIZE)[None, :] + np.arange(test.shape[0]-WINDOW_SIZE)[:, None]]
windows_test.shape

windows_validation=validation.values[np.arange(WINDOW_SIZE)[None, :] + np.arange(validation.shape[0]-WINDOW_SIZE)[:, None]]
windows_validation.shape

BATCH_SIZE =  500
N_EPOCHS = 50
hidden_size = 100

w_size=windows_train.shape[1]*windows_train.shape[2]
z_size=windows_train.shape[1]*hidden_size

windows_normal_train = windows_train[:int(np.floor(.8 *  windows_train.shape[0]))]
windows_normal_test = windows_test[int(np.floor(.8 *  windows_test.shape[0])):int(np.floor(windows_test.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_train).float().view(([windows_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_test).float().view(([windows_test.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_validation).float().view(([windows_validation.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size)
print(device)
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)
plot_history(history)

torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model.pth")


checkpoint = torch.load("model.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results=testing(model,test_loader)


windows_labels=[]
for i in range(len(labels)-WINDOW_SIZE):
    windows_labels.append(list(np.int_(labels[i:i+WINDOW_SIZE])))

y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]


y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])

threshold=ROC(y_test,y_pred)
