import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
class AudioGenerator(Dataset):
    def __init__(self, filename, database_dir_path, Xdim=(298, 257, 2), ydim=(298, 257, 2, 2), batch_size=4, shuffle=True):
        self.filename = filename
        self.Xdim = Xdim
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.database_dir_path = database_dir_path

    def __len__(self):
        return int(np.floor(len(self.filename) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        filename_temp = [self.filename[k] for k in indexes]

        X, y = self.__data_generation(filename_temp)

        return torch.Tensor(X), torch.Tensor(y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filename))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filename_temp):
        X = np.empty((self.batch_size, *self.Xdim))
        y = np.empty((self.batch_size, *self.ydim))

        for i, ID in enumerate(filename_temp):
            info = ID.strip().split(' ')
            X[i,] = np.load(self.database_dir_path+'/mix/' + info[0])

            for j in range(2):
                y[i, :, :, :, j] = np.load(self.database_dir_path+'/crm/' + info[j + 1])

        assert y[:,:,:,0] != y[:,:,:,1]
        return X, y


class AVGenerator(Dataset):
    def __init__(self, filename, database_dir_path, X1dim=(298, 257, 2), X2dim=(75, 1, 1792, 2), ydim=(298, 257, 2, 2), batch_size=4, shuffle=True):
        self.filename = filename
        self.X1dim = X1dim
        self.X2dim = X2dim
        self.ydim = ydim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.database_dir_path = database_dir_path

    def __len__(self):
        return int(np.floor(len(self.filename) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        filename_temp = [self.filename[k] for k in indexes]

        [X1, X2], y = self.__data_generation(filename_temp)

        return torch.Tensor([X1, X2]), torch.Tensor(y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.filename))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filename_temp):
        X1 = np.empty((self.batch_size, *self.X1dim))
        X2 = np.empty((self.batch_size, *self.X2dim))
        y = np.empty((self.batch_size, *self.ydim))

        for i, ID in enumerate(filename_temp):
            info = ID.strip().split(' ')
            X1[i, ] = np.load(self.database_dir_path+'audio/AV_model_database/mix/' + info[0])
            for j in range(2):
                y[i, :, :, :, j] = np.load(self.database_dir_path+'audio/AV_model_database/crm/' + info[j + 1])
                X2[i, :, :, :, j] = np.load(self.database_dir_path+'../model/pretrain_model/face1022_emb/' + info[j + 3])

        return [X1, X2], y