"""Project Sales Forecast data getter https://github.com/dafiti-group/rnd-sales-forecast."""
import datetime
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.sensors.s3_key_sensor import S3KeySensor

tmpl_search_path = Variable.get("sql_path")

default_args = {
    'start_date': airflow.utils.dates.days_ago(0),
    'retries': 1,
    'retry_delay': timedelta(minutes=60*24*15)
}

dag = DAG(
    'sales_forecast_data',
    default_args=default_args,
    description='Project Sales Forecast data getter',
    schedule_interval="0 0 1,15 * *",
    template_searchpath=tmpl_search_path,
    dagrun_timeout=timedelta(minutes=60)
)

s3_bucket = "gfg-rnd-qa-us-east-2-rnd-sales-forecast"
redshift_adapter_image = "gcr.io/dft-rnd-kubeflow/redshift-adapter:e74d0e0"

query_file_path = f"{tmpl_search_path}sales_forecast.sql"
# query_file_path = Path(__file__).parent.absolute() / query_file_path
with open(query_file_path, 'r') as file:
    query_content = file.read()

secrets_file_path = f"{tmpl_search_path}secrets.yml"
with open(secrets_file_path, 'r') as file:
    secrets_content = file.read()

now = datetime.datetime.now()

image_command = [
    f'--query="{query_content}"',
    '--rds_host={{var.value.RDS_HOST}}',
    '--rds_user={{var.value.RDS_USER}}',
    '--rds_password={{var.value.RDS_PASSWORD}}',
    '--dump_aws_access_key_id={{var.value.DUMP_AWS_ACCESS_KEY_ID}}',
    '--dump_aws_secret_access_key={{var.value.DUMP_AWS_SECRET_ACCESS_KEY}}',
    '--remote_aws_access_key_id={{var.value.RND_AWS_ACCESS_KEY_ID}}',
    '--remote_aws_secret_access_key={{var.value.RND_AWS_SECRET_ACCESS_KEY}}',
    f'--remote_storage=s3://{s3_bucket}/{now.year}-{now.month}-{now.day}/input/data/raw',
]

# priority_weight has type int in Airflow DB, uses the maximum.
task = KubernetesPodOperator(
    image=redshift_adapter_image,
    arguments=image_command,
    labels={
        "company" : "gfg",
        "env"     : "qa",
        "group"   : "rnd",
        "role"    : "machine_learning",
        "system"  : "sales-forecast",
        "type"    : "service",
    },
    namespace='default',
    name="sales-forecast-data-test",
    task_id='sales_forecast_data',
    get_logs=True,
    dag=dag,
)

# t1 = BashOperator(
#     task_id='echo',
#     bash_command='echo test',
#     dag=dag,
#     depends_on_past=False,
#     priority_weight=2**31-1
# )
# task = PushToS3(...)

check = S3KeySensor(
   task_id='check_parquet_exists',
   bucket_key=f"s3://{s3_bucket}/input/data/raw/data.parquet",
   poke_interval=0,
   timeout=0,
   dag=dag,
)
task >> check
