import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


from .utils import clean_memory, save_model
from .configs import ModelType





def train_test_pipeline(
        cfg, 
        model, 
        device, 
        optimizer, 
        loss_func, 
        epochs, 
        train_loader, 
        test_loader, 
        results_path
    ):
    clean_memory(device)
    train_losses = []
    test_losses = []
    best_test_loss = np.inf
    for rec in tqdm(range(epochs), total=epochs, desc="Training of model"):
        epoch_loss = 0
        for data, label, id in train_loader:
            data = data.to(device)
            if cfg['model_type'] == ModelType.clss:
                label = label.type(torch.long)
            else:
                label = label.unsqueeze(1)
            
            label = label.to(device)
            output = model(data)
            loss = loss_func(output, label)


            if cfg['model_type'] == ModelType.regr and cfg['add_mae']: # add MAE loss
                loss += 0.1*torch.mean(torch.abs(output - label))

            if cfg['l2_lambda']: # L2 reg
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += cfg['l2_lambda'] * l2_reg
                loss = torch.sqrt(loss) # ?

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss / len(train_loader)
        train_losses.append(epoch_loss.item())
        
        with torch.no_grad():
            epoch_test_loss = 0
            for data, label, id in test_loader:
                data = data.to(device)
                if cfg['model_type'] == ModelType.clss:
                    label = label.type(torch.long)
                else:
                    label = label.unsqueeze(1)
                label = label.to(device)

                test_output = model(data)
                test_loss = loss_func(test_output, label)
                epoch_test_loss += test_loss / len(test_loader)
            if best_test_loss > epoch_test_loss.item():
                best_test_loss = epoch_test_loss.item()
                print(f"New model dumped by better test loss! {rec}")
                save_model(model, Path(results_path)/"best_by_test_loss.pth")
            
            test_losses.append(epoch_test_loss.item())

        print(f"Epoch {rec+1}/{epochs}:")
        print(f"Train Loss: {epoch_loss.item():.4f}")
        print(f"Test Loss: {epoch_test_loss.item():.4f}")
        print("\n")
        
    save_model(model, Path(results_path)/"final.pth")
    return train_losses, test_losses



# inference & metrics
def infer_model(model, loader, device, model_type):
    model.eval()
    reals = []
    predicted = []
    with torch.no_grad():
        for data, label, id in tqdm(loader):
            data = data.to(device)
            if model_type == ModelType.clss:
                label = label.type(torch.long)
            else:
                label = label.unsqueeze(1)
            output = model(data)

            reals += label.cpu().detach().reshape(-1).tolist()
            if model_type == ModelType.regr:
                predicted += output.cpu().detach().reshape(-1).tolist()
            else:
                predicted += output.cpu().detach().tolist()

    if model_type == ModelType.clss:
        predicted = np.array(predicted).argmax(axis=1)
    return np.array(reals), np.array(predicted)
