WITH raw_medium_data AS(
    SELECT 
        fullvisitorid AS fvid,
        TO_HEX(MD5(CONCAT(
              COALESCE(trafficSource.campaign, '_'), 
              COALESCE(trafficSource.source, '_'),
              COALESCE(trafficSource.medium, '_'), 
              COALESCE(trafficSource.keyword, '_'), 
              COALESCE(trafficSource.adContent, '_')
        ))) AS utm_hash,
        (SELECT value FROM UNNEST(customDimensions) WHERE index=8) AS hash_id,
        visitstarttime AS sess,
        IF(trafficSource.isTrueDirect= True,'(direct)',LOWER(REGEXP_REPLACE(trafficsource.source, r'[,;:]', ''))) AS mediumsrc,
        IF(trafficSource.isTrueDirect = True,'(none)',LOWER(REGEXP_REPLACE(trafficsource.medium, r'[,;:]', ''))) AS medium,
        IF(trafficSource.isTrueDirect = True,1, (IF(REGEXP_REPLACE(trafficsource.keyword, r'[,;:]', '') LIKE '%dafit%',1,0))) AS is_some_kw_dafiti,
        IF(trafficSource.isTrueDirect = True,1, (IF(REGEXP_REPLACE(trafficsource.keyword, r'[,;:]', '') LIKE 'dafiti',1,0))) AS is_kw_dafiti,
        COALESCE(totals.totalTransactionRevenue/1e6, 0) AS value,
        IF((SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='6')=0, 0, 1) AS is_conversion,
    FROM `dafiti-analytics.40663402.ga_sessions_*`
    WHERE {table_suffix_filter}
),
aggregate_by_fvid AS (
    SELECT
        fvid,
        ARRAY_AGG(STRUCT(
            hash_id,
            utm_hash,
            sess,
            medium,
            mediumsrc,
            is_some_kw_dafiti,
            is_kw_dafiti,
            is_conversion,
            IF(is_conversion=1, value, NULL) AS value
        ) ORDER BY sess) AS s
    FROM raw_medium_data
    GROUP BY fvid
)
,
create_client_id AS (
    SELECT
        ARRAY(
            SELECT AS STRUCT
                * EXCEPT(hash_id),
                COALESCE(
                    FIRST_VALUE(hash_id IGNORE NULLS) OVER (ORDER BY sess ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING), # next non-null idhash
                    LAST_VALUE(hash_id IGNORE NULLS) OVER (ORDER BY sess ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), # previous non-null idhash
                    CONCAT('v',fvid) # if there is no hash id, use fullVisitorId
                ) AS client_id
            FROM UNNEST(s)
        ) AS u
    FROM aggregate_by_fvid 
),
get_conversion_group AS (
    SELECT
        u.client_id,
        u.utm_hash,
        u.sess,
        u.medium,
        u.mediumsrc,
        u.is_some_kw_dafiti,
        u.is_kw_dafiti,
        u.value,
        COALESCE(SUM(u.is_conversion) OVER (PARTITION BY u.client_id ORDER BY u.sess ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS conv_group
    FROM create_client_id, UNNEST(u) AS u
),
agg_by_client AS (
    SELECT
        client_id,
        ARRAY_AGG(
        STRUCT(conv_group,
        utm_hash,
        sess,
        medium,
        mediumsrc,
        is_some_kw_dafiti,
        is_kw_dafiti,
        value) ORDER BY sess) AS c
    FROM get_conversion_group
    GROUP BY client_id
),
agg_by_conv_group AS (
    SELECT
        client_id,
        ARRAY(
            SELECT AS STRUCT
                STRING_AGG(utm_hash, ',' ORDER BY sess) AS utm_hash,
                STRING_AGG(CONCAT(mediumsrc,'_', medium,'_',CAST(is_some_kw_dafiti AS STRING)) ORDER BY sess) as trinomials,
                STRING_AGG(CAST(sess AS STRING), ',' ORDER BY sess) AS sess,
                COALESCE(SUM(value),-1) AS value,
            FROM UNNEST(c)
        ) AS c
        FROM agg_by_client
)
SELECT
    client_id,
    (
        SELECT
            STRING_AGG(
            CONCAT(utm_hash,':',
            trinomials, ':',
            sess,':',
            CAST(value AS STRING)))
        FROM UNNEST(c)
    ) AS agg_sessions
FROM agg_by_conv_group