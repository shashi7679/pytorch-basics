
import torch
import torchvision
import wandb
import prepare_dataset
from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch.nn.functional as F
import utils
import time
import os

torch.backends.cudnn.benchmarks=True
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    os.makedirs("./saved")
except FileExistsError:
        # directory already exists
    pass

def train_log(result, epoch):
    loss = float(result['train_loss'])
    # where the magic happens
    wandb.log({"epoch": epoch, "train_loss": loss, "val_acc": result['val_acc'], "val_loss":result['val_loss']})
    print(f"Training Loss after " + str(epoch) + f" epochs: {loss:.3f}")

    
class CIFR10_CNN_Model(utils.Image_Classifcation_Base):
    def __init__(self):
        super().__init__()
        self.Model = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1,stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(256*4*4,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,xb):
        return self.Model(xb)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, model, train_loader, val_loader, learning_rate = 0.001):
    wandb.watch(model,log='all',log_freq=10)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    since = time.time()
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []   
        for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        train_log(result=result,epoch=epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), config.saved_path)

    
if __name__ == '__main__':


    train_data,_,classes,available_train_size = prepare_dataset.create_dataset(name='cifar10')

    wandb.login()

    config = dict(
        #Neccessary
        saved_path="./saved/cnn-model-cifar10.pt",
        lr=0.0001, 
        EPOCHS = 25,
        BATCH_SIZE = 64,
        TRAIN_VALID_SPLIT = 0.85,
    #################################################################### 
        #For Perforamce Tuning
        device=device,
        SEED = 42,
        pin_memory=True,
        num_workers=2,
        channels_last=True)
    ####################################################################

    wandb.init(project='cifar10-cnn',config=config)
    config = wandb.config

    training_data_size = int(config.TRAIN_VALID_SPLIT*available_train_size)
    validation_data_size = available_train_size - training_data_size


    print("Data Used for Training :-",training_data_size,"    Data used for Validation :- ",validation_data_size)

    train_dl,dev_dl = random_split(train_data,[training_data_size,validation_data_size])

    train_dl = DataLoader(train_dl,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory)
    dev_dl = DataLoader(dev_dl,
                        batch_size=config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory)

    train_dl = utils.DeviceDataLoader(train_dl, device)
    dev_dl = utils.DeviceDataLoader(dev_dl, device)
    
    model = CIFR10_CNN_Model()
    if config.channels_last:
        model = model.to(config.device, memory_format=torch.channels_last) #CHW --> #HWC
    else:
        model = model.to(config.device)

    fit(epochs=config.EPOCHS,model=model,train_loader=train_dl,val_loader=dev_dl,learning_rate=config.lr)

