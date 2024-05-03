from os import environ, getenv, listdir, system
from os.path import abspath, isdir
from pandas import read_sql_query
from psycopg2 import connect
from re import search, sub
from simplejson import loads
from sqlalchemy import create_engine
from subprocess import run
from time import strftime, time
from loguru import logger


def get_credentials(redshift_credentials_json_path: str, s3_credentials_json_path: str) -> (dict, dict):
    """Load Redshift and S3 Credentials from JSON files.

    Arguments:
        redshift_credentials_json_path {str} -- Path for JSON file containing credentials for access to Redshift.
        s3_credentials_json_path {str} -- Path for JSON file containing credentials for access to S3.

    Returns:
        Tuple(
            redshift_credentials {dictionary},
            s3_credentials {dictionary}
        )
    """

    logger.info('Fetching credentials... ')
    try:
        with open(redshift_credentials_json_path) as fh:
            redshift_credentials = loads(fh.read())
    except Exception:
        raise Exception(f'Error loading Redshift credentials at {redshift_credentials_json_path}')
    try:
        with open(s3_credentials_json_path) as fh:
            s3_credentials = loads(fh.read())
    except Exception:
        raise Exception(f'Error loading S3 credentials at {s3_credentials_json_path}')
    logger.info('Done!')
    return redshift_credentials, s3_credentials


def generate_query_string_with_unload(query: str, unload_path: str, s3_credentials: dict, verbose: bool = False) -> str:
    """Generate query string including commands to unload results to S3.
    Do not "CREATE TABLE" on the query!

    Arguments:
        query {str} -- Query string such as "SELECT * FROM table;"
        unload_path {str} -- S3 path to unload query results
        s3_credentials {dict} -- S3 credentials

    Keyword Arguments:
        verbose {bool} -- if verbose print resulting query (default: {False})

    Returns:
        output_query {str} -- query string containing unload clause to send results to S3 bucket
    """

    logger.info('Generating query...')
    query = query.rstrip(';')
    unload_permission = f'aws_access_key_id={s3_credentials["AWS_ACCESS_KEY_ID"]};aws_secret_access_key={s3_credentials["AWS_SECRET_ACCESS_KEY"]}'

    output_query = f"""
        DROP TABLE IF EXISTS #temp_rnd_redshift_unload_script;
        CREATE TABLE #temp_rnd_redshift_unload_script AS ({query});
        UNLOAD('SELECT * FROM #temp_rnd_redshift_unload_script')
        TO
        '{unload_path}'
        CREDENTIALS
        '{unload_permission}'
        DELIMITER ','
        GZIP
        ADDQUOTES
        ALLOWOVERWRITE
        ESCAPE
        PARALLEL ON;
        DROP TABLE #temp_rnd_redshift_unload_script;
    """
    logger.info('Done!')
    if(verbose):
        logger.info(output_query)
    return output_query


def redshift_execute_query(query: str, redshift_credentials: dict) -> None:
    """Connect to Redshift and execute query.

    Arguments:
        query {str} -- Query string containing unload clause
        redshift_credentials {dict} -- Redshift credentials
    """

    logger.info('Starting connection...')
    con = connect(
        dbname=redshift_credentials['db_name'],
        host=redshift_credentials['host_name'],
        port=redshift_credentials['port_num'],
        user=redshift_credentials['user_name'],
        password=redshift_credentials['password'],
    )
    cur = con.cursor()
    logger.info('Done!')
    logger.info('Executing query...')
    cur.execute(query)
    cur.close()
    con.close()
    logger.info('Done!')


def redshift_create_header(query: str, redshift_credentials: dict, output_file: str) -> None:
    """Perform a query with 'LIMIT 0' to get columns names and create a CSV file containing only a header.

    Arguments:
        query {str} -- Query string NOT containing unload clause
        redshift_credentials {dict} -- Redshift credentials
        output_file {str} -- Path for output file
    """

    logger.info('Fetching columns names and creating output file...')
    engine = create_engine(
        f"postgresql://{redshift_credentials['user_name']}" +
        f":{redshift_credentials['password']}@{redshift_credentials['host_name']}" +
        f":{redshift_credentials['port_num']}/{redshift_credentials['db_name']}"
    )

    query = sub(r'--.*[\n]', '', query)
    query = sub(r'[\n]+', ' ', query)
    query = sub(r'\/\*.*?\*\/', '', query)
    query = sub(r'[\s]+', ' ', query).strip(' ').strip(';')
    query = sub(r'[Ll][Ii][Mm][Ii][Tt] [\d]+$', '', query)

    read_sql_query(query+' LIMIT 0;', engine).to_csv(output_file, index=False)
    logger.info('Done!')


def download_data_from_s3(s3_credentials: str, unload_path: str, export_path: str, verbose: bool = True) -> str:
    """Download data from s3. Downloads everything from s3://<unload_path>/ to
        <export_path>/TEMP__<unload_path innermost folder>.

    Arguments:
        s3_credentials {str} -- S3 credentials
        unload_path {str} -- Unload path
        export_path {str} -- Export path

    Keyword Arguments:
        verbose {bool} -- if verbose print stdout from aws bash command (default: {True})

    Returns:
        temp_export_path {str} -- Path for folder containing downloaded files.
    """

    logger.info('Downloading files from s3...')
    temp_export_path = f'{export_path.rstrip("/")}/TEMP__{unload_path.strip("/").split("/")[-1]}/'
    environ['AWS_ACCESS_KEY_ID'] = s3_credentials["AWS_ACCESS_KEY_ID"]
    environ['AWS_SECRET_ACCESS_KEY'] = s3_credentials["AWS_SECRET_ACCESS_KEY"]
    download_output = run(
        (f'aws s3 cp {unload_path} {temp_export_path} --recursive').split(),
        capture_output=True)
    if(verbose):
        logger.info(download_output.stdout.decode())
    return temp_export_path


def convert_to_csv(gz_files_folder: str, output_file: str) -> None:
    """Convert .gz files in folder and append to existing CSV file (or create new file).

    Arguments:
        gz_files_folder {str} -- Path of folder containing .gz files
        output_file {str} -- Path for output CSV file
    """

    logger.info('Converting files...')
    system((f'zcat {gz_files_folder}*.gz>>{output_file}'))
    logger.info(f'Done! Created file: {output_file}')


def delete_temp_folder(gz_files_folder: str) -> None:
    """Check if folder contains only files in pattern '####_part_##.gz' and delete it
    If folder contains other files, delete only '####_part_##.gz' files    <======== [NOT IMPLEMENTED]

    Arguments:
        gz_files_folder {str} -- Path of folder containing .gz files (default: {None})
    """

    list_of_files = listdir(gz_files_folder)
    pattern = r'^[0-9]{4}_part_[0-9]{2}.gz$'
    files_not_in_pattern = [f for f in list_of_files if (search(pattern, f) is None)]
    if(len(files_not_in_pattern) == 0):
        logger.info(f'Deleting folder {gz_files_folder}...')
        system((f'rm -rf {gz_files_folder}'))
        logger.info('Done!')
    else:
        logger.info(f'Warning! The following files inside the folder do not match the pattern "{pattern}":')
        [logger.info(f'>    {f}') for f in files_not_in_pattern]
        logger.info(f'NOT DELETING FOLDER {gz_files_folder}')
        files_in_pattern = [f for f in list_of_files if (search(pattern, f) is not None)]
        for f in files_in_pattern:
            logger.info(f'Deleting: {gz_files_folder.rstrip("/")}/{f}')
            system((f'rm {gz_files_folder.rstrip("/")}/{f}'))
            logger.info('Done!')


def get_csv(
        query: str,
        export_path: str = './',
        name: str = None,
        redshift_credentials_json_path: str = None,
        s3_credentials_json_path: str = None,
        bucket_name: str = 'gfg-rnd-dev-us-east-1-dump',
        midpath: str = 'dafit-br-dump/rnd_redshift_getcsv',
        ) -> str:
    """Performs a query against Redshift, unload results to S3, downloads the results from S3,
    converts it to a CSV file with header, and removes temporary files (.gz files).

    Arguments:
        query {str} -- Query string without "unload" clause (e.g.: query="SELECT * FROM table")

    Keyword Arguments:
        export_path {str} -- Path to export the CSV file to (default: {'./'})
        name {str} -- query name (not a path, not a filename) (default: {unnamed_<YYYYMMDD_HHMMSS>})
        redshift_credentials_json_path {str} -- Path for JSON file containing credentials
            for access to Redshift (default: {<HOME>/.credentials/redshift_credentials.json})
        s3_credentials_json_path {str} -- Path for JSON file containing credentials
            for access to S3 (default: {<HOME>/.credentials/s3_credentials.json})
        bucket_name {str} -- Bucket name (default: 'gfg-rnd-dev-us-east-1-dump/dafit-br-dump')
        midpath {str} -- midpath to unload results to S3. unload_path will be
            s3://<bucket_name>/<midpath>/<name>__<posix> (default: 'rnd_redshift_getcsv')

    Returns:
        output_file {str} -- Path of output CSV file
    """

    if redshift_credentials_json_path is None:
        redshift_credentials_json_path = f'{getenv("HOME").rstrip("/")}/.credentials/redshift_credentials.json'
    if s3_credentials_json_path is None:
        s3_credentials_json_path = f'{getenv("HOME").rstrip("/")}/.credentials/s3_credentials.json'

    if name is None:
        name = f'unnamed_{strftime("%Y%m%d_%H%M%S")}'
    elif '/' in name:
        raise Exception('name cannot contain "/"')
    midpath = midpath.strip('/')
    export_path = abspath(export_path).rstrip('/')+'/'
    if not isdir(export_path):
        raise Exception(f'The folder {export_path} does not exist.')
    posix = f'{time():.0f}'
    unload_path = f's3://{bucket_name.lstrip("s3://").rstrip("/")}/{midpath}/{name}__{posix}/'
    output_file = f'{export_path}{name}.csv'

    rs_creds, s3_creds = get_credentials(
        redshift_credentials_json_path,
        s3_credentials_json_path,
    )
    sql = generate_query_string_with_unload(
        query=query,
        unload_path=unload_path,
        s3_credentials=s3_creds,
        verbose=False,
    )
    redshift_execute_query(
        query=sql,
        redshift_credentials=rs_creds,
    )
    temp_export_path = download_data_from_s3(
        s3_credentials=s3_creds,
        unload_path=unload_path,
        export_path=export_path,
    )
    redshift_create_header(
        query,
        redshift_credentials=rs_creds,
        output_file=output_file,
    )
    convert_to_csv(temp_export_path, output_file=output_file)
    delete_temp_folder(temp_export_path)
    return output_file