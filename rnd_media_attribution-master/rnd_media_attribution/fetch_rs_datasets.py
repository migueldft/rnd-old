from utils import redshift as rs
from yaml import safe_load
from loguru import logger
import os


config_file = './config.yml'


def main():
    with open(config_file, mode='r') as f:
        config = safe_load(f)
    with open(config['etl']['redshift_sql_file'], mode='r') as f:
        query = f.read()

    csv_path = os.path.abspath(os.path.split(config['etl']['redshift_csv_filename'])[0])
    csv_filename = os.path.splitext(os.path.split(config['etl']['redshift_csv_filename'])[1])[0]
    if not os.path.isdir(csv_path):
        logger.info(f'Creating directory {csv_path}')
        os.makedirs(csv_path, exist_ok=True)

    rs.get_csv(
        query = query,
        export_path = csv_path,
        name = csv_filename,
        bucket_name = config['etl']['s3_bucket']
    )

if __name__ == '__main__':
    main()
    