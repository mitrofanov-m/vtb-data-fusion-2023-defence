import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold
from .utils import custom_read_csv
import json
from tqdm import tqdm

def convert_different_inputs_to_df(method_input) -> pd.DataFrame:
    if isinstance(method_input, str):
        object = pd.read_csv(method_input)
    elif isinstance(method_input, pd.DataFrame):
        object = method_input
    return object

def validate_results(result_clear, 
                   result_fraud, 
                   targets, n_splits=5) -> float:
    
    result_clear = convert_different_inputs_to_df(result_clear)
    result_fraud = convert_different_inputs_to_df(result_fraud)
    targets = convert_different_inputs_to_df(targets)
    result = result_clear.merge(result_fraud, how='inner', left_on='user_id', right_on='user_id')
    result = result.merge(targets, how='left', left_on='user_id', right_on='user_id')
    mean_harm_roc_aucs = []

    roc_auc_original = roc_auc_score(result.target, result.clear_proba)
    roc_auc_attacked = roc_auc_score(result.target, result.fraud_proba)
    roc_auc = 2 / (1 / roc_auc_original  + 1 / roc_auc_attacked)
    return roc_auc
    # skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    # for i, (_, test_index) in enumerate(skf.split(result, result.target)):
    #     test = result.iloc[test_index]
    #     roc_auc_original = roc_auc_score(test.target, test.clear_proba)
    #     roc_auc_attacked = roc_auc_score(test.target, test.fraud_proba)
    #     roc_auc = 2 / (1 / roc_auc_original  + 1 / roc_auc_attacked)
    #     print(f'Validation iter={i+1}, roc_auc={roc_auc}')
    #     mean_harm_roc_aucs.append(roc_auc)
    
    # print('-------------------- result -------------------- ')
    # print(f'mean_harm_roc_auc ={np.mean(mean_harm_roc_aucs)}')
    # return np.mean(mean_harm_roc_aucs)


def validate_model_(predict_method: object, validation_df_path: str, bins_path: str, 
                    model_path: str, targets_path: str, quantiles_path: str, random_seed: int=20230206, n_splits=5) -> float:
    BUDGET = 10

    # загрузка необходимых данных
    targets = pd.read_csv(targets_path)
    with open(quantiles_path, 'r') as f:
        quantiles = json.load(f)
    df_transactions = custom_read_csv(validation_df_path)
    # получение скоров
    result_clear = predict_method(validation_df_path, bins_path, model_path, random_seed=random_seed)
    threshold = result_clear.target.max() / 2 
    poor_user = result_clear.user_id.loc[result_clear.target.argmin()]
    hero_user = result_clear.user_id.loc[result_clear.target.argmax()]


    one_idx = result_clear.index[result_clear.target > threshold]  # Эти пользователи похожи на Героя
    zero_idx = result_clear.index[result_clear.target <= threshold] # А эти на Неудачника

    users = result_clear.user_id.values
    one_users = users[one_idx]
    zero_users = users[zero_idx]

    for user in tqdm(users):
        if user in one_users:
            copy_from = poor_user # похожим на Героя скопируем 10 последних транзакций Неудачника
        else:
            copy_from = hero_user # А похожим на Неудачника наоборот

        idx_to = df_transactions.index[df_transactions.user_id == user][-BUDGET:]
        idx_from = df_transactions.index[df_transactions.user_id == copy_from][-BUDGET:]
        sign_to = np.sign(df_transactions.loc[idx_to, "transaction_amt"].values)
        sign_from = np.sign(df_transactions.loc[idx_from, "transaction_amt"].values)
        sign_mask = (sign_to == sign_from)
        df_transactions.loc[idx_to[sign_mask], "mcc_code"] = df_transactions.loc[idx_from[sign_mask], "mcc_code"].values
        df_transactions.loc[idx_to[sign_mask], "transaction_amt"] = df_transactions.loc[idx_from[sign_mask], "transaction_amt"].values
    
    fraud_df = df_transactions
    result_fraud = predict_method(fraud_df, bins_path, model_path, random_seed=random_seed)

    result_clear = result_clear.rename(columns={'target': 'clear_proba'})
    result_fraud = result_fraud.rename(columns={'target': 'fraud_proba'})
    
    mean_harm_roc_auc = validate_results(result_clear, result_fraud, targets, n_splits=n_splits)
    return mean_harm_roc_auc

def validate_model(predict_method: object, clear_df_path: str, 
                   fraud_df_path: str, bins_path: str, model_path: str, targets_path: str, random_seed=20230206, n_splits=5) -> float:
    # загрузка выбранных методов
    targets = pd.read_csv(targets_path)
    # получение скоров
    result_clear = predict_method(clear_df_path, bins_path, model_path, random_seed=random_seed)
    result_fraud = predict_method(fraud_df_path, bins_path, model_path, random_seed=random_seed)

    result_clear = result_clear.rename(columns={'target': 'clear_proba'})
    result_fraud = result_fraud.rename(columns={'target': 'fraud_proba'})
    
    mean_harm_roc_auc = validate_results(result_clear, result_fraud, targets, n_splits=n_splits)
    return mean_harm_roc_auc