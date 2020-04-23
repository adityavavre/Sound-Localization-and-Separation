import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class AV_model(nn.Module):
    def __init__(self, people_num=2):
        super(AV_model, self).__init__()
        audio_lst = []
        audio_lst.append(nn.Conv2d(2, 96, kernel_size=(1, 7), stride=(1, 1), dilation=1, padding=(0, 3)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(7, 1), stride=(1, 1), dilation=1, padding=(3, 0)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=1, padding=(2, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(2, 1), padding=(4, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(4, 1), padding=(8, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(8, 1), padding=(16, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(16, 1), padding=(32, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(32, 1), padding=(64, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(1, 1), padding=(2, 2)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(2, 2), padding=(4, 4)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(4, 4), padding=(8, 8)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(8, 8), padding=(16, 16)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(16, 16), padding=(32, 32)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), dilation=(32, 32), padding=(64, 64)))
        audio_lst.append(nn.BatchNorm2d(96))
        audio_lst.append(nn.ReLU())
        audio_lst.append(nn.Conv2d(96, 8, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), padding=(0, 0)))
        audio_lst.append(nn.BatchNorm2d(8))
        audio_lst.append(nn.ReLU())
        self.AS = nn.Sequential(*audio_lst)
        video_lst = []
        video_lst.append(nn.Conv2d(2, 256, kernel_size=(7, 1), stride=(1, 1), dilation=(1, 1), padding=(3, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        video_lst.append(nn.Conv2d(256, 256, kernel_size=(5, 1), stride=(1, 1), dilation=(1, 1), padding=(2, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        video_lst.append(nn.Conv2d(256, 256, kernel_size=(5, 1), stride=(1, 1), dilation=(2, 1), padding=(4, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        video_lst.append(nn.Conv2d(256, 256, kernel_size=(5, 1), stride=(1, 1), dilation=(4, 1), padding=(8, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        video_lst.append(nn.Conv2d(256, 256, kernel_size=(5, 1), stride=(1, 1), dilation=(8, 1), padding=(16, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        video_lst.append(nn.Conv2d(256, 256, kernel_size=(5, 1), stride=(1, 1), dilation=(16, 1), padding=(32, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        video_lst.append(nn.Conv2d(256, 256, kernel_size=(5, 1), stride=(1, 1), dilation=(16, 1), padding=(32, 0)))
        video_lst.append(nn.BatchNorm2d(256))
        video_lst.append(nn.ReLU())
        self.VS = nn.Sequential(*video_lst) 
        self.upsample = self.UpSampling2DBilinear((298, 256))
        self.BiLSTM = nn.LSTM(input_size=298, hidden_size=400, bidirectional=True)
        self.fc1 = nn.Linear(400, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.complex_mask = nn.Linear(600, 257*2*people_num)


    def UpSampling2DBilinear(self, size):
        return nn.Upsample(size=size, mode='bilinear', align_corners=True)

    def forward(self, audio_input, video_input):
        batch_size, timesteps, C, H, W = video_input.shape 
        audio_out = self.AS(audio_input.permute(0, 3, 2, 1))
        audio_out = audio_out.view((batch_size, -1, 298, 8*257))
        

        AVfusion_list = [audio_out]
        for i in range(people_num):
            video_out = self.VS(video_input[:, :, :, i])
            video_out = video_out.view((batch_size, -1, 75, 256, 1))
            video_out = self.upsample(video_out)
            video_out = video_out.view((batch_size, -1, 298, 256))    
            AVfusion_list.append(video_out)
        AVfusion = torch.cat(AVfusion_list, 3)
        AVfusion = AVfusion.view((batch_size*timesteps, C, H, W))
        AVfusion = self.BiLSTM(AVfusion)
        AVfusion = self.fc1(AVfusion)
        AVfusion = F.relu(AVfusion)
        AVfusion = self.fc2(AVfusion)
        AVfusion = F.relu(AVfusion)
        AVfusion = self.fc3(AVfusion)
        AVfusion = F.relu(AVfusion)
        AVfusion = self.complex_mask(AVfusion)
        return AVfusion.view((-1, 298, 257, 2, self.people_num))