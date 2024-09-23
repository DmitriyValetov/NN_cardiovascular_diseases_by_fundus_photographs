import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
from operator import itemgetter


from core.data import load_data
from core.pipelines import train_test_pipeline, infer_model
from core.configs import tgt2class_num, tgt2Modeltype, model2resize2batchsize, ModelType
from core.utils import recreate_dir, data_targets_cut, dump_json
from core.data import get_loaders
from core.models import initialize_model
from core.plotting import plot_train_process, display_results
from core.metrics import get_metrics




def make_StratifiedKFold_exp(cfg):
    stop_signal = False
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

    
    skf = StratifiedKFold(n_splits=cfg['n_fold'], shuffle=True, random_state=cfg['random_state'])
    for i, (train_index, test_index) in enumerate(skf.split(data, targets)):
        print(f"Fold {i}:")
        cfg[f'fold'] = i
        recreate_dir(Path(cfg['results_path'])/f"fold_{i}")
        
        train_data, test_data = itemgetter(*list(train_index))(data), itemgetter(*list(test_index))(data)
        train_loader, test_loader = get_loaders(train_data, test_data, cfg['batch_size'], cfg['resize_size'], cfg['model_type'])

        tmp = cfg.copy()
        del tmp['model_type']
        dump_json(tmp, Path(cfg['results_path'])/f"fold_{i}"/"cfg.json")

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
                Path(cfg['results_path'])/f"fold_{i}",
            )
            plot_train_process(cfg['target'], train_loss, test_loss, Path(cfg['results_path'])/f"fold_{i}")
        
        except KeyboardInterrupt:
            stop_signal = True
            print("Doing last inference for visualizing")



        if cfg['model_name'] == ModelType.regr:
            model = torch.load(Path(cfg['results_path'])/f"fold_{i}"/"best_by_test_loss.pth")
        else:
            model = torch.load(Path(cfg['results_path'])/f"fold_{i}"/"final.pth") # this is awkward for clss

        reals, predicted = infer_model(model, train_loader, cfg['device'], cfg['model_type'])
        display_results(reals, predicted, cfg['model_name'], cfg['model_type'], "train", Path(cfg['results_path'])/f"fold_{i}", list(range(cfg['class_num'])))

        reals, predicted = infer_model(model, test_loader, cfg['device'], cfg['model_type'])
        display_results(reals, predicted, cfg['model_name'], cfg['model_type'], "test", Path(cfg['results_path'])/f"fold_{i}", list(range(cfg['class_num'])))

        results = get_metrics(reals, predicted, cfg['model_type'], labels=list(range(cfg['class_num'])))
        dump_json(results, Path(cfg['results_path'])/f"fold_{i}"/f"best_loss_model_results.json")

        if stop_signal:
            exit(0)





if __name__ == "__main__":
    regr = []
    clss = []

    regr = ['age']
    clss = ['sex']
    for model_name in ['efficientnet-b0',  'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3']:
    # for model_name in ['resnet18']:
    # for model_name in ['densenet121']:
        for target in regr+clss:
            for resize_size in [256, 512]:
                for lr in [0.0001, 0.00005]:
                    for l2_lambda in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
                        print("Start exp for ", model_name, target, resize_size, lr, l2_lambda)
                        cfg = {
                            "n_fold": 5,
                            'dataset': "nii_gb",
                            # 'dataset': "eyes_v1",

                            'resize_size': resize_size,
                            'model_name': model_name,

                            'demo': True,
                            'demo_N': 100,
                            'epochs': 3,

                            # 'demo': False,
                            # 'epochs': 300,

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

                        make_StratifiedKFold_exp(cfg)

