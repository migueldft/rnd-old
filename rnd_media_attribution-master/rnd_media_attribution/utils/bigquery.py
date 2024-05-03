from google.cloud import bigquery as bq, storage
from subprocess import run
import pandas as pd
import time
import datetime
import pytz
import os
import re
from loguru import logger

_STANDARD_GCP_credentials_json_path = os.path.join(os.getenv("HOME"),'.credentials/bigquery_credentials.json')




def create_table_from_csv(
            filename:str,
            dataset_id:str,
            table_id:str,
            gcp_credentials_json_path:str=None,
            allow_overwrite:bool=False
    ) -> bq.LoadJob.result:
    """Create table in BigQuery from local CSV file.
    
    Arguments:
        filename {str} -- Path to CSV file
        dataset_id {str} -- BigQuery dataset ID in which table will be created
        table_id {str} -- BigQuery table ID for created table
    
    Keyword Arguments:
        gcp_credentials_json_path {str} -- Path for JSON file containing credentials for access to BigQuery (default: {<HOME>/.credentials/bigquery_credentials.json})
        allow_overwrite {bool} -- If allow_overwrite is FALSE, overwriting tables in BigQuery is allowed (default: {True})
    
    Returns:
        bq.LoadJob.result -- Load job result
    """

    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    logger.info('Starting BigQuery client... ')
    client = bq.Client()
    logger.info('Done!')
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    job_config = bq.LoadJobConfig()
    job_config.source_format = bq.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True

    if allow_overwrite:
        job_config.write_disposition = bq.WriteDisposition.WRITE_TRUNCATE
    else:
        job_config.write_disposition = bq.WriteDisposition.WRITE_EMPTY

    logger.info(f'Creating {dataset_id}:{table_id} from {filename}... ')
    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    logger.info(f"Loaded {job.output_bytes} bytes into {dataset_id}:{table_id}.")
    return job.result()




def append_to_table_from_csv(
            filename:str,
            dataset_id:str,
            table_id:str,
            gcp_credentials_json_path:str=None
    ) -> bq.LoadJob.result:
    """Append data to an existing table in BigQuery from local CSV file.
    
    Arguments:
        filename {str} -- Path to CSV file
        dataset_id {str} -- BigQuery dataset ID in which table will be created
        table_id {str} -- BigQuery table ID for created table
    
    Keyword Arguments:
        gcp_credentials_json_path {str} -- Path for JSON file containing credentials for access to BigQuery (default: {<HOME>/.credentials/bigquery_credentials.json})
    
    Returns:
        bq.LoadJob.result -- Load job result
    """

    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    logger.info('Starting BigQuery client... ')
    client = bq.Client()
    logger.info('Done!')
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bq.LoadJobConfig()
    job_config.source_format = bq.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True
    job_config.write_disposition = bq.WriteDisposition.WRITE_APPEND
    logger.info(f'Appending data to {dataset_id}:{table_id} from {filename}... ')
    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    logger.info(f"Loaded {job.output_bytes} rows into {dataset_id}:{table_id}.")
    return job.result()




def set_table_expiration_countdown(
            dataset_id:str,
            table_id:str,
            weeks_to_expire_table:int=0,
            days_to_expire_table:int=0,
            hours_to_expire_table:int=0,
            minutes_to_expire_table:int=0,
            gcp_credentials_json_path:str=None
    ) -> bq.table:
    """Set the time in which an existing BigQuery table will expire.
    
    Arguments:
        dataset_id {str} -- BigQuery dataset ID containing the table for which the expiration date will be set
        table_id {str} -- BigQuery table ID of table for which the expiration date will be set
    
    Keyword Arguments:
        weeks_to_expire_table {int} -- Weeks counting from the moment of execution to expire BigQuery table (default: {0})
        days_to_expire_table {int} -- Days counting from the moment of execution to expire BigQuery table (default: {0})
        hours_to_expire_table {int} -- Hours counting from the moment of execution to expire BigQuery table (default: {0})
        minutes_to_expire_table {int} -- Minutes counting from the moment of execution to expire BigQuery table (default: {0})
        gcp_credentials_json_path {str} -- Path for JSON file containing credentials for access to BigQuery (default: {<HOME>/.credentials/bigquery_credentials.json})
    
    Returns:
        bq.table -- Table that had expiration date modified
    """

    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    logger.info('Starting BigQuery client... ')
    client = bq.Client()
    logger.info('Done!')
    table_obj = client.get_table(client.dataset(dataset_id).table(table_id))
    expires_at = datetime.datetime.now(pytz.utc) + datetime.timedelta(
        weeks=weeks_to_expire_table,
        days=days_to_expire_table,
        hours=hours_to_expire_table,
        minutes=minutes_to_expire_table)
    table_obj.expires = expires_at
    logger.info('Updating table expiration date... ')
    logger.info(f'(Table: {client.project}:{dataset_id}.{table_id})')
    table = client.update_table(table_obj,["expires"])
    logger.info('Done!')
    newexp = client.get_table(client.dataset(dataset_id).table(table_id)).expires
    logger.info(f'New expiration date: ', end='')
    logger.info(newexp.astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S (UTC%z)'))
    return table




def unset_table_expiration(
            dataset_id:str,
            table_id:str,
            gcp_credentials_json_path:str=None
    ) -> bq.table:
    """Set the expiration date of an existing BigQuery table to "never".
    
    Arguments:
        dataset_id {str} -- BigQuery dataset ID containing the table for which the expiration date will be set
        table_id {str} -- BigQuery table ID of table for which the expiration date will be set
    
    Keyword Arguments:
        gcp_credentials_json_path {str} -- Path for JSON file containing credentials for access to BigQuery (default: {<HOME>/.credentials/bigquery_credentials.json})
    
    Returns:
        bq.table -- Table that had expiration date modified
    """

    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    logger.info('Starting BigQuery client... ')
    client = bq.Client()
    logger.info('Done!')
    table_obj = client.get_table(client.dataset(dataset_id).table(table_id))
    table_obj.expires = None
    logger.info('Updating table expiration date... ')
    logger.info(f'(Table: {client.project}:{dataset_id}.{table_id})')
    table = client.update_table(table_obj,["expires"])
    logger.info('Done!')
    return table




def query_against_bigquery(
            query:str,
            destination_dataset_id:str=None,
            destination_table_id:str=None,
            gcp_credentials_json_path:str=None,
            check_before_query:bool=True,
            allow_overwrite:bool=False,
            MAX_BYTES_BILLED:int=1073741824
    ) -> bq.QueryJob.result:
    """Executes a query against BigQuery.
    
    Arguments:
        query {str} -- String of query to be executed against BigQuery, e.g. query="SELECT column FROM `project_id:dataset_id.table_id`"
    
    Keyword Arguments:
        destination_dataset_id {str} -- BigQuery dataset ID in which query results will be written (default: {None})
        destination_table_id {str} -- BigQuery table ID in which query results will be written (default: {None})
        gcp_credentials_json_path {str} -- Path for JSON file containing credentials for access to BigQuery (default: {<HOME>/.credentials/bigquery_credentials.json})
        check_before_query {bool} -- if(check_before_query), a dry-run is executed to check for errors and calculate amount of data that will be processed (default: {True})
        allow_overwrite {bool} -- if(allow_overwrite), results can overwrite existing tables
        MAX_BYTES_BILLED {int} -- maximum number of bytes to be billed on query against BigQuery (default: 1073741824 [=1 GB])
    Returns:
        bq.QueryJob.result -- Query job result
    """

    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    logger.info('Starting BigQuery client... ')
    client = bq.Client()
    logger.info('Done!')
    if check_before_query:
        logger.info('Performing dry run... ')
        job_config_dryrun = bq.QueryJobConfig()
        job_config_dryrun.maximum_bytes_billed = int(MAX_BYTES_BILLED)
        job_config_dryrun.dry_run=True
        job_config_dryrun.use_query_cache = False
        if allow_overwrite:
            job_config_dryrun.write_disposition = bq.WriteDisposition.WRITE_TRUNCATE
        else:
            job_config_dryrun.write_disposition = bq.WriteDisposition.WRITE_EMPTY
        query_job_dryrun = client.query(query,job_config=job_config_dryrun)
        assert query_job_dryrun.state == "DONE"
        assert query_job_dryrun.dry_run
        query_size = get_metric_prefix(query_job_dryrun.total_bytes_processed,1024)
        logger.info(f"This query will process {query_size[0]:.1f} {query_size[1]}B.")
        for i in range(10,0,-1):
            logger.info(f'Starting query in {i} seconds...')
            time.sleep(1)
    logger.info('Running query')
    job_config = bq.QueryJobConfig()
    job_config.maximum_bytes_billed = int(MAX_BYTES_BILLED)
    job_config.dry_run=False
    job_config.use_query_cache = False
    if allow_overwrite:
        job_config.write_disposition = bq.WriteDisposition.WRITE_TRUNCATE
    else:
        job_config.write_disposition = bq.WriteDisposition.WRITE_EMPTY
    if (destination_dataset_id is not None) and (destination_table_id is not None):
        job_config.destination = client.dataset(destination_dataset_id).table(destination_table_id)
    query_job = client.query(query,job_config=job_config)
    results = query_job.result()
    logger.info('Done!')
    return results




def get_metric_prefix (value:float, base:int) -> (float, str):
    assert ((base==1000) | (base==1024))
    metric_prefixes = {0:'', 1:'k', 2:'M', 3:'G', 4:'T', 5:'P', 6:'E', 7:'Z', 8:'Y'}
    order = 8
    for n in range(1,len(metric_prefixes)):
        base=1024
        if(base**n > value):
            order = n - 1
            break
    x = value/(base**order)
    return x, metric_prefixes[order]



def table_extract(
            dataset_id:str,
            table_id:str,
            gcs_bucket:str,
            destination_format:str='CSV', #-------------------------------------------------------------------------------####################################
            gcp_credentials_json_path:str=None
    ) -> str:
    assert destination_format.upper() in ('CSV','JSON','AVRO'), 'destination_format.upper() must be (\'CSV\',\'JSON\',\'AVRO\')'
    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    datetime_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    posix_string = str(round(time.time()))
    destination = os.path.join(dataset_id, table_id, datetime_string, f'PART_{posix_string}_*.csv.gz')
    destination_uri = os.path.join("gs://", gcs_bucket, destination)
    logger.info('Starting BigQuery client... ')
    client = bq.Client()
    logger.info('Done!')
    logger.info('Getting table information... ')
    table_ref = client.dataset(dataset_id).table(table_id)
    logger.info('Done!')
    extract_job_config = bq.ExtractJobConfig()
    extract_job_config.compression = bq.job.Compression.GZIP
    format_dictionary = {
        'CSV': bq.job.DestinationFormat.CSV,
        'JSON': bq.job.DestinationFormat.NEWLINE_DELIMITED_JSON,
        'AVRO': bq.job.DestinationFormat.AVRO
    }
    extract_job_config.destination_format = format_dictionary[destination_format.upper()]
    extract_job_config.print_header = False
    logger.info('Starting extraction job... ')
    extract_job = client.extract_table(table_ref,destination_uri,job_config=extract_job_config)
    logger.info('Waiting for extraction job to complete...')
    extract_job_result = extract_job.result()
    assert extract_job_result.state=="DONE"
    logger.info('Done!')
    return destination




def download_data_from_storage(
            gcs_bucket:str,
            object_pattern:str,
            download_path:str,
            gcp_credentials_json_path:str=None
    ) -> list:

    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    real_download_path = os.path.realpath(download_path)
    storage_cli = storage.Client()
    bucket_ref = storage_cli.get_bucket(gcs_bucket)
    blobs_list = list(bucket_ref.list_blobs())
    search_pattern = object_pattern.replace("*","[0-9]+")
    logger.info(search_pattern)
    blobs_list_filtered = [blob for blob in blobs_list if (re.search(f'^{search_pattern}$', blob.name) is not None)]
    if len(blobs_list_filtered)==0:
        logger.error('No objects where found')
        raise Exception('No objects where found')
    else:
        logger.info(f'Found {len(blobs_list_filtered)} objects')
        run(["mkdir","-p",real_download_path],capture_output=True)
    
    logger.info('Downloading files from Google Cloud Storage...',flush=True)
    files = [None]*len(blobs_list_filtered)
    for i,blob in enumerate(blobs_list_filtered):
        filename = os.path.join(real_download_path,os.path.split(blob.name)[-1])
        files[i] = filename
        logger.info(f'--- {i+1}/{len(blobs_list_filtered)} ---\nDownloading object {blob.name}\nfrom {blob.bucket}...',flush=True)
        blob.download_to_filename(filename)
        logger.info(f'Downloaded to {filename}',flush=True)
    logger.info('--------------------')
    return files




def get_schema(
            dataset_id:str,
            table_id:str,
            gcp_credentials_json_path:str=None
    ) -> list:
    if gcp_credentials_json_path is None:
        gcp_credentials_json_path = _STANDARD_GCP_credentials_json_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_json_path
    print('Getting table schema...')
    client = bq.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    schema = client.get_table(table_ref).schema
    print('Done')
    return schema




def merge_gz_files_to_single_csv(
            csv_gz_files:list,
            csv_filename:str,
            schema:list
    ):
    print('Creating csv file with header')
    columns = [column.name for column in schema]
    header = ','.join(columns)
    with open(csv_filename,"w") as ftemp:
        print(header,file=ftemp)
    with open(csv_filename,"a") as ftemp:
        for i,f in enumerate(csv_gz_files):
            print(f'Converting file {i+1}/{len(csv_gz_files)}:  ', os.path.split(f)[-1])
            run(['zcat',f], stdout=ftemp)
    print('Done!')