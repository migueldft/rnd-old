from utils import bigquery as bq
from loguru import logger
from yaml import safe_load
from subprocess import run
import os

config_file = './config.yml'




def generate_search_query(config):
    with open(config['etl']['bigquery_sql_file']) as f:
        query = f.read()
    def generate_table_suffix_filter(config:dict) -> str:
        date_ranges = config['etl']['query_date_ranges']
        table_suffix_filter = 'FALSE\n'
        for r in date_ranges:
            if hasattr(r,'__iter__'):
                assert len(r) in (1,2), "len(config['etl']['query_date_ranges'][i]) must be 1 or 2"
            else:
                r=[r]
            for d in r:
                if type(d) != int:
                    raise ValueError(f'Invalid value {d} in config > etl > query_date_rage. Values must be of type int.')
            r = [(d if d>0 else 99999999) for d in r]
            table_suffix_filter += f"OR _TABLE_SUFFIX BETWEEN '{min(r):08d}' AND '{max(r):08d}'\n"
            table_suffix_filter += f"OR _TABLE_SUFFIX BETWEEN 'intraday_{min(r):08d}' AND 'intraday_{max(r):08d}'\n"
        return table_suffix_filter
    
    # return query.format(table_suffix_filter = generate_table_suffix_filter(config))
    return query.replace("{table_suffix_filter}", generate_table_suffix_filter(config))


def set_expiration(config):
    expiration_keys = ('weeks_to_expire_table', 'days_to_expire_table', 'hours_to_expire_table', 'minutes_to_expire_table')
    bigquery_table_expiration = config['etl'].get('bigquery_table_expiration', {})
    for key in bigquery_table_expiration.keys():
        assert key in expiration_keys, f'Unexpected key "{key}" in config > etl > bigquery_table_expiration. Keys must be in {expiration_keys}'
    for key in expiration_keys:
        exp_value = bigquery_table_expiration.get(key, 0)
        bigquery_table_expiration[key] = exp_value if exp_value is not None else 0
    if all(v==0 for v in bigquery_table_expiration.values()):
        logger.info('Table will not expire')
        return
    else:
        bq.set_table_expiration_countdown(
            dataset_id=config['etl']['bigquery_destination_dataset'],
            table_id=config['etl']['bigquery_destination_table'],
            weeks_to_expire_table = bigquery_table_expiration['weeks_to_expire_table'],
            days_to_expire_table = bigquery_table_expiration['days_to_expire_table'],
            hours_to_expire_table = bigquery_table_expiration['hours_to_expire_table'],
            minutes_to_expire_table = bigquery_table_expiration['minutes_to_expire_table']
        )


def main():
    with open(config_file, mode='r') as f:
        config = safe_load(f)

    query = generate_search_query(config)
    logger.info('Generated query:')
    logger.info(query)

    csv_path = os.path.abspath(os.path.split(config['etl']['bigquery_csv_filename'])[0])
    if not os.path.isdir(csv_path):
        logger.info(f'Creating directory {csv_path}')
        os.makedirs(csv_path, exist_ok=True)

    bq.query_against_bigquery(
        query=query,
        destination_dataset_id=config['etl']['bigquery_destination_dataset'],
        destination_table_id=config['etl']['bigquery_destination_table'],
        MAX_BYTES_BILLED=int(1073741824 * config['etl']['MAX_GIGABYTES_BILLED']),
        allow_overwrite=True
    )

    set_expiration(config)

    gcs_destination = bq.table_extract(
        dataset_id=config['etl']['bigquery_destination_dataset'],
        table_id=config['etl']['bigquery_destination_table'],
        gcs_bucket=config['etl']['gcs_bucket']
    )

    downloaded_temp_files = bq.download_data_from_storage(
        gcs_bucket = config['etl']['gcs_bucket'],
        object_pattern = gcs_destination,
        download_path = './'
    )

    schema = bq.get_schema(
        dataset_id = config['etl']['bigquery_destination_dataset'],
        table_id = config['etl']['bigquery_destination_table']
    )

    bq.merge_gz_files_to_single_csv(
        csv_gz_files = downloaded_temp_files,
        csv_filename = config['etl']['bigquery_csv_filename'],
        schema = schema
    )

    logger.info('Deleting temporary files...')
    for temp_file in downloaded_temp_files:
        try:
            run(['rm', temp_file], capture_output=True)
            logger.info(f'File {temp_file} deleted')
        except Exception as e:
            logger.warning('An exception occured when trying to delete the file {temp_file}')
            logger.warning(type(e))
            logger.warning(e.args)
            logger.warning(e)



if __name__ == '__main__':
    main()

