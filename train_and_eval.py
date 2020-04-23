import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import model_AV as AV
from DataGen import AVGenerator
import os
from model_loss import audio_discriminate_loss as audio_loss

# create AV model
#############################################################
RESTORE = False
# If set true, continue training from last checkpoint
# needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

# super parameters
people_num = 2
num_epoch = 10
initial_epoch = 0
batch_size = 1 # 4 to feed one 16G GPU
gamma_loss = 0.1
beta_loss = gamma_loss*2

# physical devices option to accelerate training process
workers = 1 # num of core
use_multiprocessing = False
NUM_GPU = 0

# PATH
path = './saved_AV_models' # model path
database_dir_path = '../../data/'

# create folder to save models
folder = os.path.exists(path)
if not folder:
    os.makedirs(path)
    print('create folder to save models')
filepath = path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.h5"

# read train and val file name
# format: mix.npy single.npy single.npy
trainfile = []
valfile = []
with open((database_dir_path+'AV_log/AVdataset_train.txt'), 'r') as t:
    trainfile = t.readlines()
with open((database_dir_path+'AV_log/AVdataset_val.txt'), 'r') as v:
    valfile = v.readlines()
AV_model = AV.AV_model(people_num)

train_loader = DataLoader(AVGenerator(trainfile,database_dir_path= database_dir_path, batch_size=batch_size, shuffle=True))
val_loader = DataLoader(AVGenerator(valfile,database_dir_path=database_dir_path, batch_size=batch_size, shuffle=True))

optimizer = torch.optim.Adam(AV_model.parameters(), lr=1e-4)
lossfunc = audio_loss(gamma=gamma_loss, num_speaker=people_num)
for epoch in range(0, num_epoch):
    print(epoch)
    for batch in train_loader:
        preds = AV_model(Variable(batch[0]))
        loss = lossfunc(preds, Variable(batch[1]))
        print(loss)
        loss.backward()
        optimizer.step()
    if epoch%10 == 0:   
        torch.save(AV_model.state_dict(), str(epoch)+'.pt')  

import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils

model_path = './saved_AV_models/AVmodel-2p-150-0.28740.h5'
dir_path = './pred/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
database_path = '../../data/audio/AV_model_database/mix/'
face_path = '../../model/pretrain_model/face1022_emb/'

testfiles = []
with open('../../data/AV_log/AVdataset_train.txt', 'r') as f:
    testfiles = f.readlines()

def parse_X_data(line,num_people=people_num,database_path=database_path,face_path=face_path):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(num_people):
        single_idxs.append(names[i])
    file_path = database_path + mix_str
    mix = np.load(file_path)
    face_embs = np.zeros((1,75,1,1792,num_people))
    for i in range(num_people):
        face_embs[0,:,:,:,i] = np.load(face_path+"%05d_face_emb.npy"%int(single_idxs[i]))

    return mix,single_idxs,face_embs


loss = audio_loss(gamma=0.1, num_speaker=2)
AV_model.load_state_dict(torch.load(model_path))
for line in testfiles:
    mix, single_idxs, face_embs = parse_X_data(line)
    mix_expand = np.expand_dims(mix, axis=0)
    cRMs = AV_model([mix_expand, face_embs])
    cRMs = cRMs[0]
    prefix = ""
    for idx in single_idxs:
        prefix += idx + "-"
    for i in range(people_num):
        cRM = cRMs[:,:,:,i]
        assert cRM.shape == (298,257,2)
        F = utils.fast_icRM(mix,cRM)
        T = utils.fast_istft(F,power=False)
        filename = dir_path+prefix+single_idxs[i]+'.wav'
        wavfile.write(filename,16000,T)