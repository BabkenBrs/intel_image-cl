import time
import torchvision
from tqdm import tqdm
from data import ImageFolderCustom, init_dataloader
from timeit import default_timer as timer

import numpy as np
import torch

# from intel_image_cl.data import ImageFolderCustom, init_dataloader

def train_model(model, train_loader, val_loader, schedul_learning, loss_fn, opt, device, n_epochs: int):
    """Training the model

    Args:
        model: model to train
        train_loader: dataloader for train data
        val_loader (torch.nn.utils.Dataloader): dataloader for validation data
        loss_fn: loss function to be used
        opt: optimizer to be used
        device: device used for training
        n_epochs: number of epochs to train
    """
    st_time = timer()

    valid_loss_min = np.inf
    Valid_loss = []
    Train_loss = []

    for epoch in tqdm(range(1,n_epochs+1)):

        
        train_loss  = 0.0 
        valid_loss  = 0.0 

        #train
        model.train()
        
        for batch , (X, y) in enumerate(train_loader):       
        
            
            X, y = X.to(device), y.to(device)
        
            opt.zero_grad()
        
            y_pred = model(X)
        
            loss = loss_fn(y_pred , y)
            
            loss.backward()
            
            opt.step()
            
            
            train_loss += loss.item()*X.size(0)
        

    
        #validation
        model.eval()
        
        for batch , (X, y) in enumerate(val_loader):
                
            X , y = X.to(device), y.to(device)
        
            y_pred = model(X)
        
            loss = loss_fn(y_pred, y)
        
            valid_loss += loss.item()*X.size(0)
    
            
        
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(val_loader.sampler)
        Valid_loss.append(valid_loss)
        Train_loss.append(train_loss)
    
        schedul_learning.step()
        schedul_learning
        end_time = timer()
    
        print('Epoch: {} \nTraining Loss: {:.3f} \nValidation Loss: {:.3f}'.format(epoch, train_loss, valid_loss))
        
    
        if valid_loss <= valid_loss_min:
            
            print("Decrease Validation Loss {:.4f} : {:.4f} ".format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'My_Model.pt')
            valid_loss_min = valid_loss
            
        print('Learning Rate : {:.4f}'.format(opt.state_dict()['param_groups'][0]['lr']))
        print('\nTime : {:.2f}'.format(end_time - st_time))
        print('----------------------------------')

    return Train_loss, Valid_loss


def main():
    NUM_CLASSES=6

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # model = SimpleClassifier()
    Model = torchvision.models.resnet18(pretrained=True)
    for param in Model.parameters():
        param.required_grad = False
    number_feature = Model.fc.in_features
    Model.fc = torch.nn.Linear(in_features=number_feature , out_features=NUM_CLASSES)

    opt = torch.optim.SGD(params=Model.parameters() , lr = 0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function.to(device)
    schedul_learning = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt , milestones=[3 , 6 ] ,
                                                        gamma=0.055)

    data_dir = "/Users/babkenbrsikyan/Documents/Babken/MLOps/intel_image_cl/data"
    train_dataset = ImageFolderCustom(data_dir + "/seg_train", train_transform=True)
    val_dataset = ImageFolderCustom(data_dir + "/seg_test", train_transform=False)

    train_loader = init_dataloader(train_dataset, 128)
    val_loader = init_dataloader(val_dataset, 128)

    train_loss, val_loss = train_model(
        Model, train_loader, val_loader, schedul_learning, loss_function, opt, device, 3
    )


if __name__ == "__main__":
    main()