
UNLOAD('SELECT t.sku_config AS sku_config, t.product_medium_image_url AS product_medium_image_url FROM #table_1 as t GROUP BY t.product_medium_image_url, t.sku_config;')
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