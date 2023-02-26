import numpy as np
import pandas as pd
import pickle
import pytorch_lightning as pl
import torch
import numpy as np

from .model import (
    TransactionsRnn,
    TransactionsDataset,
    process_for_nn,
    get_dataloader
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_predict(source_file, bins_path, model_path, random_seed=20230206):
    """
        Для защиты от заменны транзакций мы выберем случайным образом 90% данных по каждому пользователю, предскажем его класс оригинальной моделью.
        Сделаем так 9 раз и усредним. Таким образом мы уменьшим влияение измененных транзакций - их всего 10 из 300, и случайная подвыборка
        и усреднение должны размыть их влияние. См. метод reliable_predict
    """
    REPETITIONS = 30  # Сколько повторений делаем
    pl.seed_everything(random_seed)
    if isinstance(source_file, pd.DataFrame):
        df_transactions = source_file
    else:
        df_transactions = (
            pd.read_csv(
                source_file,
                parse_dates=["transaction_dttm"],
                dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
            )
            .dropna()
            .assign(
                hour=lambda x: x.transaction_dttm.dt.hour,
                day=lambda x: x.transaction_dttm.dt.dayofweek,
                month=lambda x: x.transaction_dttm.dt.month,
                number_day=lambda x: x.transaction_dttm.dt.day,
            )
        )

    with open(bins_path, "rb") as f:
        bins = pickle.load(f)

    features = bins.pop("features")

    model = TransactionsRnn()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    results = []
    for i in range(REPETITIONS):
        # Сэмплируем данные. Каждый раз - разный random_seed
        df = process_for_nn(df_transactions.sample(frac=0.98, random_state=random_seed+i, replace=True), features, bins)
        dataset = TransactionsDataset(df)
        dataloader = get_dataloader(dataset, device, is_validation=True)
        preds = []
        users = []
        for data, target in dataloader:
            y_pred = model(data)
            preds.append(y_pred.detach().cpu().numpy())
            users.append(target.detach().cpu().numpy())
        preds = np.concatenate(preds)
        users = np.concatenate(users)
        results.append(pd.DataFrame({"user_id": users, "target": preds[:, 1]}))
    results[0]['target'] = np.median([x.target.values for x in results], axis=0) # усредняем предсказания
    return results[0]


def reliable_predict(source_file, bins_path, model_path, random_seed=20230206):
    """
        Для защиты от заменны транзакций мы выберем случайным образом 90% данных по каждому пользователю, предскажем его класс оригинальной моделью.
        Сделаем так 9 раз и усредним. Таким образом мы уменьшим влияение измененных транзакций - их всего 10 из 300, и случайная подвыборка
        и усреднение должны размыть их влияние. См. метод reliable_predict
    """
    REPETITIONS = 9  # Сколько повторений делаем
    pl.seed_everything(random_seed)
    if isinstance(source_file, pd.DataFrame):
        df_transactions = source_file
    else:
        df_transactions = (
            pd.read_csv(
                source_file,
                parse_dates=["transaction_dttm"],
                dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
            )
            .dropna()
            .assign(
                hour=lambda x: x.transaction_dttm.dt.hour,
                day=lambda x: x.transaction_dttm.dt.dayofweek,
                month=lambda x: x.transaction_dttm.dt.month,
                number_day=lambda x: x.transaction_dttm.dt.day,
            )
        )

    with open(bins_path, "rb") as f:
        bins = pickle.load(f)

    features = bins.pop("features")

    model = TransactionsRnn()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    results = []
    for i in range(REPETITIONS):
        # Сэмплируем данные. Каждый раз - разный random_seed
        df = process_for_nn(df_transactions.sample(frac=0.9, random_state=random_seed+i, replace=True), features, bins)
        dataset = TransactionsDataset(df)
        dataloader = get_dataloader(dataset, device, is_validation=True)
        preds = []
        users = []
        for data, target in dataloader:
            y_pred = model(data)
            preds.append(y_pred.detach().cpu().numpy())
            users.append(target.detach().cpu().numpy())
        preds = np.concatenate(preds)
        users = np.concatenate(users)
        results.append(pd.DataFrame({"user_id": users, "target": preds[:, 1]}))
    results[0]['target'] = np.mean([x.target.values for x in results], axis=0) # усредняем предсказания
    return results[0]

def predict(source_file, bins_path, model_path, random_seed=20230206):
    pl.seed_everything(random_seed)
    if isinstance(source_file, pd.DataFrame):
        df_transactions = source_file
    else:
        df_transactions = (
            pd.read_csv(
                source_file,
                parse_dates=["transaction_dttm"],
                dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
            )
            .dropna()
            .assign(
                hour=lambda x: x.transaction_dttm.dt.hour,
                day=lambda x: x.transaction_dttm.dt.dayofweek,
                month=lambda x: x.transaction_dttm.dt.month,
                number_day=lambda x: x.transaction_dttm.dt.day,
            )
        )

    with open(bins_path, "rb") as f:
        bins = pickle.load(f)

    features = bins.pop("features")
    df = process_for_nn(df_transactions, features, bins)
    dataset = TransactionsDataset(df)
    dataloader = get_dataloader(dataset, device, is_validation=True)

    model = TransactionsRnn()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    preds = []
    users = []
    for data, target in dataloader:
        y_pred = model(data)
        preds.append(y_pred.detach().cpu().numpy())
        users.append(target.detach().cpu().numpy())
    preds = np.concatenate(preds)
    users = np.concatenate(users)
    return pd.DataFrame({"user_id": users, "target": preds[:, 1]})
