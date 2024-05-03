UNLOAD('SELECT * FROM #table_1;')
    TO
    's3://{bucket}/{prefix}'
    -- IAM_ROLE
    -- 'aws_iam_role={iam_role}'
    CREDENTIALS
    'aws_access_key_id={aws_access_key_id};aws_secret_access_key={aws_secret_access_key};token={aws_session_token}'
    DELIMITER ';'
    GZIP
    ADDQUOTES
    ALLOWOVERWRITE
    ESCAPE
PARALLEL ON;