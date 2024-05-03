import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
import multiprocessing

from .logger import set_up_logging


logger = set_up_logging(__name__)


def download(
        file_path: str,
        bucket: str,
        remote_path: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str
    ) -> bool:
    try:
        s3 = boto3.client('s3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        config = TransferConfig(
            max_concurrency=multiprocessing.cpu_count())
        logger.info(f"Download {remote_path} to {file_path}")
        with open(file_path, 'wb') as data:
            s3.download_fileobj(bucket, remote_path, data, Config=config)
    except ClientError as e:
        logger.error(e)
        return False
    return True
        

def upload(
        file_path: str,
        bucket: str,
        remote_path: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str
    ) -> bool:
    try:
        s3 = boto3.client('s3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        response = s3.upload_file(
            file_path, bucket, remote_path)
        logger.info(response)
    except ClientError as e:
        logger.error(e)
        return False
    return True


def list_files(
        bucket: str,
        prefix: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str
    ) -> list:
    files = []
    try:
        s3 = boto3.client('s3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        kwargs = {'Bucket': bucket, 'Prefix': prefix}
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp["Contents"]:
            files.append(obj["Key"])
    except ClientError as e:
        logger.error(e)
        return False
    return files


def sync_data(
        local_path: str,
        bucket: str,
        prefix: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str,
        **kwargs: dict
    ) -> None:
    files = list_files(bucket, prefix,
        aws_access_key_id,
        aws_secret_access_key,
        aws_session_token)

    # return files
    if files:
        for f in files:
            file_name = f.split('/')[-1]
            download(
                f'{local_path}/{file_name}',
                bucket, f,
                aws_access_key_id,
                aws_secret_access_key,
                aws_session_token
            )