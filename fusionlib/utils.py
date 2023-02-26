import pandas as pd

def custom_read_csv(source_file: str):
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

    return df_transactions