import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from Agent import Agent
agent = Agent(n_heads=16).to(device)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/loss')


class dataset(Dataset):
    def __init__(self):
        columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
        self.metadata = pd.read_csv("./sim-track-data/driving_log.csv", names=columns)
        
        self.transform = torchvision.transforms.Resize((256,256))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        obs_path = self.metadata.iloc[idx]["center"]
        steer = self.metadata.iloc[idx]["steering"]
        throttle = self.metadata.iloc[idx]["throttle"]
        reverse = self.metadata.iloc[idx]["reverse"]
        
        obs = torchvision.io.read_image("./sim-track-data/IMG/"+os.path.basename(obs_path)).float()
        obs = self.transform(obs).to(device)
        return obs, [steer, throttle, reverse]
    

my_dataset = dataset()
train_set, val_set = torch.utils.data.random_split(my_dataset, [len(my_dataset)*0.8, len(my_dataset)*0.2])

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


lr = 1e-2

brain_optimizer = torch.optim.Adam(agent.get_brain_parameters(), lr=lr*1e-2, 
                    betas=(0.92, 0.999))
loss_fn = nn.MSELoss()


def train_loop(epoch):
    size = len(dataloader)

    train_loss = 0
    for batch, (observations, label) in enumerate(train_loader):
        
        label = torch.cat([x.float() for x in label])
        
        agent.attention_model.prev_Q = torch.zeros(16, 256, 16, 16).to(device) 

        task_memory_optimizer = torch.optim.Adam(agent.get_task_memory_parameters(task), lr=lr, 
                            betas=(0.92, 0.999))

        loss = 0
        for i in range(batch_size):
            pred_action_dist = agent(observations[i], task)
            loss += loss_fn(pred_action_dist, label[i])
        

        train_loss += loss

        brain_optimizer.zero_grad()
        task_memory_optimizer.zero_grad()

        loss.backward()

        brain_optimizer.step()
        task_memory_optimizer.step()

        if batch % 100 == 0:
            writer.add_scalar('train_loss',
                train_loss / len(train_loss),
                batch + size*epoch)

            print(f"loss: {train_loss:>7f}  [Epoch: {epoch}; {batch:>5d}/{size:>5d}]")

            val_loss = 0
            for batch, (observations, label) in enumerate(val_loader):
                for i in range(batch_size):
                    pred_action_dist = agent(observations[i], task)
                    val_loss += loss_fn(pred_action_dist, label[i])

            writer.add_scalar('val_loss',
                val_loss / len(val_loss),
                batch + size*epoch)            


epochs = 300
task = "driving"
for epoch in range(epochs):
    train_loop(epoch)
    agent.save_parameters()


