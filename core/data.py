import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import numpy as np
from torchvision import transforms


from .configs import ModelType




def load_data(df, ds_path, target, stratification_name='age_group'):
    data = []
    stratification_values = []
    pats_without_row = []
    pats_with_row = []
    pats_empty_dir = []
    df_pats = set(df.pat.values)
    for pat in os.listdir(ds_path): # (root,dirs,files) in os.walk(domain_path):
        pat_path = Path(ds_path)/pat
        files = os.listdir(pat_path)
        files = [f for f in files if f.lower().endswith('jpg') or f.lower().endswith('png') or f.lower().endswith('jpeg')]
        if len(files)>0:
            if not pat in df_pats:
                pats_without_row.append(pat)
            else:
                pats_with_row.append(pat)
                row = df[df.pat==pat].iloc[0]
                data.append([
                    Path(ds_path)/pat,
                    row[target],
                    [Path(ds_path)/pat/f for f in files],
                ])
                if stratification_name:
                    stratification_values.append(row[stratification_name])
        else:
            pats_empty_dir.append(pat)
            
    return data, stratification_values


class dataset(Dataset):
    def __init__(self, data, transform=None, target_type=ModelType.regr):
        self.data = data
        self.transform = transform
        self.target_type = target_type
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        root, target, imgs_paths = self.data[idx]
        try:
            img = Image.open(random.choice(imgs_paths)).convert('RGB')
        except Exception as e:
            print(f"Error loading image for {root}: {str(e)}")
            raise e
        img_transformed = self.transform(img)
        if self.target_type == ModelType.regr:
            return img_transformed, np.float32(target), ""
        return img_transformed, np.int32(target), ""


def get_loaders(train_data, test_data, batch_size, resize_size, target_type=ModelType.regr):
    train_transforms =  transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            # transforms.RandomResizedCrop(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    test_transforms = transforms.Compose([   
            transforms.Resize((resize_size, resize_size)),
            # transforms.RandomResizedCrop(resize_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    train_ds = dataset(train_data, transform=train_transforms, target_type=target_type)
    test_ds = dataset(test_data, transform=test_transforms, target_type=target_type)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader