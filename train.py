import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.optim as optim
import torch.nn as nn


from core.data import load_data
from core.pipelines import train_test_pipeline, infer_model
from core.configs import tgt2class_num, tgt2Modeltype, model2resize2batchsize, ModelType
from core.utils import recreate_dir, data_targets_cut, dump_json
from core.data import get_loaders
from core.models import initialize_model
from core.plotting import plot_train_process, display_results
from core.metrics import get_metrics



if __name__ == "__main__":
    stop_signal = False
    regr = []
    clss = []

    # regr = ['age']
    # clss = ['sex']
    regr = ["cholesterol", "SBP", "GFR"]
    # clss = ['sex', 'smoking', "DR", "HR", "DM", "AF"]
    # for model_name in ['efficientnet-b0',  'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3']:
    # for model_name in ['resnet18']:
    for model_name in ['densenet121']:
    # for model_name in ['efficientnet-b3']:
    # for model_name in ['efficientnet-b2']:
        # for target in regr+clss:
        # for target in clss:
        for target in regr:
            for resize_size in [256, 512]:
            # for resize_size in [512]:
                # for lr in [0.0001, 0.00005]:
                for lr in [0.00005]:
                    # for l2_lambda in [0.005, 0.001]:
                    # for l2_lambda in [0.1, 1, 10, 100]:
                    for l2_lambda in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
                        print("Start exp for ", model_name, target, resize_size, lr, l2_lambda)
                        cfg = {
                            'dataset': "nii_gb",
                            # 'dataset': "eyes_v1",

                            'resize_size': resize_size,
                            'model_name': model_name,

                            # 'demo': True,
                            # 'demo_N': 100,
                            # 'epochs': 3,

                            'demo': False,
                            'epochs': 300,

                            'add_mae': True,
                            "l2_lambda": l2_lambda,
                            'target': target,
                            'lr': lr,
                            'test_size': 0.2,
                            'random_state': 42,
                            'device': "cuda",
                        }
                        cfg['batch_size'] = model2resize2batchsize[cfg['model_name']][cfg['resize_size']]
                        cfg['class_num'] = tgt2class_num[cfg['target']]
                        cfg['model_type'] = tgt2Modeltype[cfg['target']]

                        cfg['exps_root'] = f"paper_exps_{cfg['dataset']}_{cfg['model_name']}"
                        cfg['results_path'] = str(Path(cfg['exps_root'])/f"{cfg['target']}_lr_{str(cfg['lr'])[2:]}_rs_{cfg['resize_size']}_l2_{str(cfg['l2_lambda']).replace('.', '_')}")
                        recreate_dir(cfg['results_path'])


                        # dataset
                        base_data_path = Path("..") / "data"
                        csv_path = base_data_path / "targets_processed.csv"
                        df = pd.read_csv(csv_path)
                        df['age_group'] = pd.qcut(df['age'], q=3, labels=[1, 2, 3])
                        df = df[['pat', 'age_group', cfg['target']]].dropna()
                        if cfg['model_type'] ==  ModelType.clss:
                            df[cfg['target']] = df[cfg['target']].apply(int)
                        else:
                            df[cfg['target']] = df[cfg['target']].apply(float)

                        data, targets = load_data(df, base_data_path/cfg['dataset'], cfg['target'])
                        if cfg['demo']:
                            data, targets = data_targets_cut(data, targets, cfg['demo_N'])
                        
                        print(f"\n\n\n\nDATA size: {len(data)}\n\n\n\n")

                        cfg['data_len'] = len(data)
                        train_data, test_data = train_test_split(data, test_size=cfg['test_size'], random_state=cfg['random_state'], stratify=targets)
                        train_loader, test_loader = get_loaders(train_data, test_data, cfg['batch_size'], cfg['resize_size'], cfg['model_type'])


                        tmp = cfg.copy()
                        del tmp['model_type']
                        dump_json(tmp, Path(cfg['results_path'])/"cfg.json")

                        # model
                        model = initialize_model(cfg['model_name'], cfg['device'], cfg['class_num'])
                        optimizer = optim.Adam(model.parameters(), cfg['lr'])
                        loss_func = nn.MSELoss() if cfg['model_type'] == ModelType.regr else nn.CrossEntropyLoss()



                        try:
                            # train
                            train_loss, test_loss = train_test_pipeline(
                                cfg,
                                model,
                                cfg['device'],
                                optimizer,
                                loss_func,
                                cfg['epochs'],
                                train_loader,
                                test_loader,
                                cfg['results_path'],
                            )
                            plot_train_process(cfg['target'], train_loss, test_loss, cfg['results_path'])
                        
                        except KeyboardInterrupt:
                            stop_signal = True
                            print("Doing last inference for visualizing")



                        if cfg['model_name'] == ModelType.regr:
                            model = torch.load(Path(cfg['results_path'])/"best_by_test_loss.pth")
                        else:
                            model = torch.load(Path(cfg['results_path'])/"final.pth") # this is awkward for clss

                        reals, predicted = infer_model(model, train_loader, cfg['device'], cfg['model_type'])
                        display_results(reals, predicted, cfg['model_name'], cfg['model_type'], "train", cfg['results_path'], list(range(cfg['class_num'])))

                        reals, predicted = infer_model(model, test_loader, cfg['device'], cfg['model_type'])
                        display_results(reals, predicted, cfg['model_name'], cfg['model_type'], "test", cfg['results_path'], list(range(cfg['class_num'])))

                        results = get_metrics(reals, predicted, cfg['model_type'], labels=list(range(cfg['class_num'])))
                        dump_json(results, Path(cfg['results_path'])/f"best_loss_model_results.json")

                        if stop_signal:
                            exit(0)
