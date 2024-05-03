from utils import bigquery as bq
from loguru import logger
from yaml import safe_load
from subprocess import run
import os

config_file = './config.yml'

def main():
    with open(config_file, mode='r') as f:
        config = safe_load(f)
    bq.create_table_from_csv(
        filename = config['etl']['redshift_csv_filename'],
        dataset_id = config['etl']['bigquery_destination_dataset'],
        table_id = config['etl']['redshift_to_bigquery_destination_table'],
        allow_overwrite=True
    )

if __name__ == '__main__':
    main()