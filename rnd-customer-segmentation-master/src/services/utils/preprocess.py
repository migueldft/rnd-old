from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import constants as constants


def preproc(df, training=False):
    today = datetime.today()
    df.columns = constants.INPUT_COLUMNS 
    
    df['sale_order_store_date'] = pd.to_datetime(df['sale_order_store_date'],infer_datetime_format=True)
    
    df_grouped = df.groupby('fk_customer')
    df['next_sale'] = df_grouped['sale_order_store_date'].shift(-1)
    df['next_sale']= df['next_sale'].fillna(today)
    df['delta'] = df['next_sale'] - df['sale_order_store_date']
    
    if training:
        df['status'] = ((df['delta'] < timedelta(days=700)) & (df['next_sale'] != today))
        df['status'] = df['status'].astype(int)

    df['is_last_order'] = np.where(df['delta'] >= timedelta(days=365), True, False)

    df['lifetime_revenue'] = df_grouped['gmv'].cumsum()
    df_orders = df_grouped['sale_order_store_number'].count().reset_index()
    df_orders.rename(columns={'sale_order_store_number':'lifetime_orders'}, inplace=True)
    df = pd.merge(df, df_orders, on=['fk_customer'], how='left')

    df_grouped = df.groupby('fk_customer')
    df['12m_aux'] = df_grouped['is_last_order'].shift(1)
    df['12m_aux'].fillna(1,inplace=True)
    df['12m_aux'] *= 1
    df['12m_aux'] = df['12m_aux'].astype(int)

    df_grouped = df.groupby('fk_customer')
    df['12m_aux'] = df_grouped['12m_aux'].cumsum()

    df_grouped = df.groupby(['fk_customer', '12m_aux'], as_index=False).agg({'sale_order_store_date':max})
    df_grouped.rename(columns={"sale_order_store_date": "churn_sale_date"}, inplace=True)
    df_aux = pd.merge(df, df_grouped, on=['fk_customer','12m_aux'], how='left')
    df_aux['delta_churn_date'] = df_aux['churn_sale_date'] - df_aux['sale_order_store_date']
    df_aux = df_aux[df_aux['delta_churn_date'] <= timedelta(days=365)]

    df_grouped = df_aux.groupby(['fk_customer','12m_aux'])
    df_aux['12m_revenue'] = df_grouped['gmv'].cumsum()

    df_orders = df_grouped['sale_order_store_number'].count().reset_index()
    df_orders.rename(columns={'sale_order_store_number':'12m_orders'}, inplace=True)
    df_aux = pd.merge(df_aux, df_orders, on=['fk_customer','12m_aux'], how='left')

    df_grouped = df_aux.groupby(['fk_customer'])
    df_aux['is_reactivation_sale'] = df_grouped['is_last_order'].shift(1)
    df_aux['is_reactivation_sale'].fillna(False,inplace=True)
    df = df_aux[(df_aux['is_last_order'] == True) | (df_aux['is_reactivation_sale'] == True)]
    del df_aux, df_grouped

    df.rename(columns={'gmv':'last_gmv'}, inplace=True)

    if training:
        df = df[constants.COLUMN_NAMES + constants.LABEL_COLUMN]
    else:
        df = df[constants.COLUMN_NAMES]
        for col_name in constants.COLUMN_NAMES:
            df[col_name]   = df[col_name].astype(float)

    return df