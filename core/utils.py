import gc
import os
import torch
import psutil
from pathlib import Path
import shutil
import json
import pickle

def kill_nvidia_smi():
    for proc in psutil.process_iter():
        if proc.name() == 'nvidia-smi':
            proc.kill()

def clean_memory(device):
    if device =='cuda':
        gc.collect()
        torch.cuda.empty_cache()
        kill_nvidia_smi()

def save_model(model, fp):
    os.makedirs(Path(fp).parent, exist_ok=True)    
    # torch.save(model.state_dict(), fp)
    torch.save(model, fp)

def dump_json(data, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    with open(filename, 'r', encoding="utf-8") as f:
       return json.load(f)

def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def recreate_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def data_targets_cut(data, targets, N=20):
    print("\n\n\n\nDATA IT CUT FOR DEV REASONS, THIS IS NOT PRODUCTION CODE\n\n\n\n")
    return data[:N], targets[:N]