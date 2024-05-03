import argparse
from datetime import datetime

from dateutil.relativedelta import relativedelta

from rnd_data_libs import redshift as rs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--query_path',
        type=str,
        default='ml/input/data/query.sql',
        help="A path to a SQL query (default: 'ml/input/data/query.sql')",
    )
    parser.add_argument(
        '--dataset_output_path',
        default='ml/input/data/raw/',
        help="An ouput path to SQL query result (default: 'ml/input/data/raw/')",
    )
    parser.add_argument(
        '--dataset_name',
        default='data',
        help="A filename to SQL query output (default: 'data')",
    )

    args = parser.parse_args()

    with open(args.query_path) as fq:
        query = fq.read().strip('\n')
    today_one_year_ago = (datetime.today() - relativedelta(years=1)).strftime("%Y-%m-%d")
    query.replace("date_replacement", today_one_year_ago)
    csv_path = rs.get_csv(
        query=query,
        name='data',
        export_path=args.dataset_output_path
    )
