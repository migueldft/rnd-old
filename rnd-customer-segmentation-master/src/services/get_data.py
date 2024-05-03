from collections import OrderedDict
import csv
import json
from pathlib import Path
import tempfile

import dask.dataframe as dd
from dask.delayed import delayed
import numpy as np
import pandas as pd

import constants as constants
from utils import set_up_logging, timeit, verify_params, get_redshift_adapter, sync_data, RedshiftAdapter


logger = set_up_logging(__name__)

CHANNEL_NAME = 'training'
TRAINING_PATH = constants.INPUT_PATH / CHANNEL_NAME

STEPS = ["create_data.sql", "unload_urls.sql", "unload_data.sql"]
PREFIX = ['', 'images', 'training']
DATA_DTYPES = ['',
    OrderedDict([
        ('sku_config', 'object'),
        ('product_medium_image_url', 'object')]),
    OrderedDict([
        ('date', 'object'),
        ('customer_id', 'object'),
        ('n_orders', 'int'),
        ('customer_gender', 'object'),
        ('customer_age', 'int'),
        ('customer_created_at', 'object'),
        ('customer_first_order_paid', 'object'),
        ('channel_partner_name', 'object'),
        ('sku_config', 'object'),
        ('is_campaign', 'bool'),
        ('product_name', 'object'),
        ('product_gender', 'object'),
        ('product_color', 'object'),
        ('product_brand', 'object'),
        ('product_medium_image_url', 'object'),
        ('cmc_category_bp', 'object') ,
        ('cmc_division_bp', 'object'),
        ('cmc_business_unit_bp', 'object'),
        ('google_product_category', 'object'),
        ('product_original_price', 'float'),
        ('payment_method_name', 'object'),
        ('shipping_condition', 'object'),
        ('product_discount', 'object'),
        ('shipping_discount', 'object'),
        ('sale_value', 'float'),
        ('is_coupon', 'bool'),
        ('device_name', 'object'),
        ('platform_name', 'object'),
        ('delivery_city', 'object'),
        ('delivery_state_code', 'object'),
        ('delivery_country_region', 'object'),
        ('planning_cluster', 'object'),
        ('planning_age', 'object'),
        ('ticketrange_planning', 'object'),
        ('originalpricerange_planning', 'object')])
]
is_big = ['', False, True]


@timeit
def query(
        redshift_adapter: RedshiftAdapter,
        step: str,
        **args: dict
    ) -> None:
    
    logger.info(f"step {step} begin.")
    with open(constants.SERVICES_PATH / 'queries' / step) as fp:
        q = fp.read()
    return redshift_adapter.query(q.format(**args))


def convert_data(directory, dtype, big):
    TRAINING_PATH.mkdir(parents=True, exist_ok=True)
    dfs = [delayed(pd.read_csv)(
            f, sep=';',
            header=None,
            names=list(dtype.keys()),
            dtype=dtype,
            compression='gzip'
        ) for f in directory.glob("*gz")]
    df = dd.from_delayed(dfs)
    if big:
        df.to_parquet(str(TRAINING_PATH), engine="pyarrow")
    else:
        df = df.compute()
        df.to_csv(constants.INPUT_PATH / 'urls.gz', index=False, compression='gzip')


def task(args: dict) -> None:
    redshift_adapter = get_redshift_adapter(args)
    try:
        redshift_adapter.init_con()
        orig_prefix = args.pop('prefix')
        for step, prefix, dtype, big in zip(STEPS, PREFIX, DATA_DTYPES, is_big):
            args["prefix"] = orig_prefix + prefix + '/'
            query(redshift_adapter, step=step, **args)
            if dtype:
                with tempfile.TemporaryDirectory(prefix=str(constants.ML_PREFIX / 'tmp/')) as temp_directory:
                    temp_directory = Path(temp_directory)
                    sync_data(temp_directory, **args)
                    convert_data(temp_directory, dtype, big)
    except Exception as error:
        logger.info(f'Error during redshift query: {error}')
    finally:
        redshift_adapter.close_con()

    
if __name__ == "__main__":

    with open(constants.SECRETS_PATH, 'r') as tc:
        args = json.load(tc)

    required_fields = ["bucket", "prefix", "aws_access_key_id", "aws_secret_access_key", "aws_session_token"]
    verify_params(args, required_fields, 'query')

    logger.info('STARTING QUERY.PY...')
    task(args)
    logger.info("...QUERY.PY HAS ENDED.\n")
