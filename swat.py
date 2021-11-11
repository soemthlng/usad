import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from utils import *
#from usad import *
from usad_backup import *
from sklearn.metrics import *
from pathlib import Path
from TaPR_pkg import etapr
from sklearn.decomposition import PCA
from sklearn import preprocessing

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

#Read data
# SWaT
normal = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv")#, nrows=1000)
normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
#normal.shape

# HAI
#normal_dataset = sorted([x for x in Path('input_HAI/').glob("train*.csv")])
#normal = dataframe_from_csvs(normal_dataset)
#normal = normal.drop(["time", "P1_B2004", "P1_B2016", "P1_B4002", "P1_B4022", "P1_FCV02Z", "P1_PCV02D", "P1_PP01AD", "P1_PP01AR", "P1_PP01BD", "P1_PP01BR", "P1_PP02D", "P1_PP02R", "P1_STSP", "P2_ASD", "P2_AutoGO", "P2_Emerg", "P2_ManualGO", "P2_OnOff", "P2_RTR", "P2_TripEx", "P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04", "P3_LH", "P3_LL", "P4_HT_FD", "P4_ST_FD", "attack", "attack_P1", "attack_P2", "attack_P3"], axis=1)
#normal = normal.drop(["time", "C02", "C03", "C06", "C08", "C09", "C10", "C11", "C14", "C18", "C19", "C21", "C22", "C33", "C34", "C35", "C37", "C40", "C43", "C45", "C50", "C51", "C53", "C59", "C61", "C63", "C64", "C65", "C67"], axis=1)

# Transform all columns into float64
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)

#from sklearn import preprocessing
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=demian7607&logNo=222009975984
#min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.StandardScaler()
x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
#normal = pd.DataFrame(x_scaled)
#normal = pd.DataFrame(x)

# PCA
pca = PCA(n_components=51)
printcipalComponents = pca.fit_transform(x_scaled)
normal= pd.DataFrame(data=printcipalComponents)



#Read data
#SWaT
attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv",sep=";")#, nrows=1000)
attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
#attack.shape

#HAI
#attack_dataset = sorted([x for x in Path('input_HAI/').glob("test*.csv")])
#attack = dataframe_from_csvs(attack_dataset)
#attack = attack.drop(["time", "attack_P1", "attack_P2", "attack_P3"], axis=1)
#attack = attack.drop(["time"], axis=1)
#attack = attack.drop(["time", "C02", "C03", "C06", "C08", "C09", "C10", "C11", "C14", "C18", "C19", "C21", "C22", "C33", "C34", "C35", "C37", "C40", "C43", "C45", "C50", "C51", "C53", "C59", "C61", "C63", "C64", "C65", "C67"], axis=1)



# Transform all columns into float64
for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)


x = attack.values 
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)
#attack = pd.DataFrame(x)

# PCA
pca = PCA(n_components=51)
printcipalComponents = pca.fit_transform(x_scaled)
attack = pd.DataFrame(data=printcipalComponents)


window_size=8


windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
#print("window normal shape", windows_normal.shape)
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
#print("window attack shape", windows_attack.shape)

import torch.utils.data as data_utils

BATCH_SIZE =  7919
N_EPOCHS = 300
hidden_size = 18

w_size=windows_normal.shape[1]*windows_normal.shape[2]
z_size=windows_normal.shape[1]*hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 * .5 * windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 * .5 * windows_normal.shape[0])):int(np.floor(.5 * windows_normal.shape[0]))]
windows_normal_test = windows_normal[int(np.floor(.5 * windows_normal.shape[0])):]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(np.concatenate([windows_normal_test,windows_attack])).float().view(([windows_normal_test.shape[0]+windows_attack.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = UsadModel(w_size, z_size)
#print(model) see model info
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)

plot_history(history)


torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model_swat.pth")



# after model load, re-train HAI and test
checkpoint = torch.load("model_swat.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results=testing(model,test_loader)

y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])
threshold = 1
# threshold = 0.9 this is best
#y_pred_ = np.zeros(y_pred.shape[0])
#y_pred_[y_pred >= threshold] = 1
y_test=np.concatenate([np.zeros(windows_normal_test.shape[0]),
                       np.ones(windows_attack.shape[0])])
#print("normal_test\n", windows_normal_test.shape[0], "attack\n", windows_attack.shape[0])

#histogram(y_test,y_pred)

#threshold=ROC(y_test,y_pred)

#confusion_matrix(y_test, np.where(y_pred > threshold, 1, 0))
#confusion_matrix(y_test,np.where(y_pred > threshold, 1, 0),perc=True)
#print("threshold : ", threshold)
test_labels = put_labels(y_pred, threshold)
print("recall", recall_score(y_test,test_labels))
print("precision", precision_score(y_test, test_labels))
print("f1", f1_score(y_test, test_labels))


### HAI



TaPR = etapr.evaluate_haicon(anomalies=y_test, predictions=test_labels)
print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")





















