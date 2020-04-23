# create custom loss function for training

import torch

def audio_discriminate_loss(gamma=0.1,num_speaker=2):
    def loss_func1(S_true,S_pred,gamma=gamma,num_speaker=num_speaker):
        sum = 0
        for i in range(num_speaker):
            sum += torch.sum(torch.flatten((torch.pow(S_true[:,:,:,i]-S_pred[:,:,:,i], 2))))
            for j in range(num_speaker):
                if i != j:
                    sum -= gamma*torch.sum(torch.flatten((torch.pow(S_true[:,:,:,i]-S_pred[:,:,:,j], 2))))

        loss = sum / (num_speaker*298*257*2)
        return loss
    loss_func1.__name__ = "lmao1"    
    return loss_func1

# loss = audio_discriminate_loss()
# l = loss(torch.randn((3,2,3,2)), torch.randn(3,2,3,2))
# print(l)