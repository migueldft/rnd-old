WITH raw_medium_data AS(
    SELECT 
        fullvisitorid AS fvid,
        (SELECT value FROM UNNEST(customDimensions) WHERE index=8) AS hash_id,
        visitstarttime AS sess,
        LOWER(REGEXP_REPLACE(trafficsource.medium, r'[,;:]', '')) AS medium,
        LOWER(REGEXP_REPLACE(trafficsource.source, r'[,;:]', '')) AS mediumsrc,
        IF(REGEXP_REPLACE(trafficsource.keyword, r'[,;:]', '') LIKE '%dafiti%',1,0) AS is_some_kw_dafiti,
        IF(REGEXP_REPLACE(trafficsource.keyword, r'[,;:]', '') LIKE 'dafiti',1,0) AS is_kw_dafiti,
        device.deviceCategory AS device,
        COALESCE(totals.totalTransactionRevenue/1e6, 0) AS value,
        (SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='1') AS click_trought_product_views,
        (SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='2') AS n_views, 
        IF((SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='6')=0, 0, 1) AS is_conversion,
        IF((SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='3')=0, 0, 1) AS is_to_cart,
        IF((SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='4')=0, 0, 1) AS is_rm_cart,
    FROM `dafiti-analytics.40663402.ga_sessions_*`
    WHERE {table_suffix_filter}
),
aggregate_by_fvid AS (
    SELECT
        fvid,
        ARRAY_AGG(STRUCT(
            hash_id,
            sess,
            medium,
            mediumsrc,
            is_some_kw_dafiti,
            is_kw_dafiti,
            device,
            click_trought_product_views,
            n_views,
            is_to_cart,
            is_rm_cart,
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
        u.sess,
        u.medium,
        u.mediumsrc,
        u.is_some_kw_dafiti,
        u.is_kw_dafiti,
        u.device,
        u.click_trought_product_views,
        u.n_views,
        u.is_to_cart,
        u.is_rm_cart,
        u.value,
        COALESCE(SUM(u.is_conversion) OVER (PARTITION BY u.client_id ORDER BY u.sess ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS conv_group
    FROM create_client_id, UNNEST(u) AS u
),
agg_by_client AS (
    SELECT
        client_id,
        ARRAY_AGG(STRUCT(conv_group,sess,medium,mediumsrc,is_some_kw_dafiti,is_kw_dafiti,device,click_trought_product_views,n_views,is_to_cart,is_rm_cart,value) ORDER BY sess) AS c
    FROM get_conversion_group
    GROUP BY client_id
),
agg_by_conv_group AS (
    SELECT
        client_id,
        ARRAY(
            SELECT AS STRUCT
                conv_group,
                STRING_AGG(medium, ',' ORDER BY sess) AS media,
                STRING_AGG(mediumsrc, ',' ORDER BY sess) AS sources,
                STRING_AGG(CAST(is_some_kw_dafiti AS STRING), ',' ORDER BY sess) AS is_some_kw_dafiti,
                STRING_AGG(CAST(is_kw_dafiti AS STRING), ',' ORDER BY sess) AS is_kw_dafiti,
                STRING_AGG(CAST(sess AS STRING), ',' ORDER BY sess) AS sess,
                STRING_AGG(device, ',' ORDER BY sess) AS device,
                STRING_AGG(CAST(click_trought_product_views AS STRING),',' ORDER BY sess) AS click_trought_product_views,
                STRING_AGG(CAST(n_views AS STRING),',' ORDER BY sess) AS views,
                STRING_AGG(CAST(is_to_cart AS STRING),',' ORDER BY sess) AS to_cart,
                STRING_AGG(CAST(is_rm_cart AS STRING),',' ORDER BY sess) AS rm_cart,
                COALESCE(SUM(value),-1) AS value,
            FROM UNNEST(c)
            GROUP BY conv_group
        ) AS c
        FROM agg_by_client
)
SELECT
    client_id,
    (
        SELECT
            STRING_AGG( CONCAT(sources,':',media,':',is_some_kw_dafiti,':',is_kw_dafiti,':',sess,':',device,':',click_trought_product_views,':', views,':',to_cart,':',rm_cart,':', CAST(value AS STRING)) , ';order,'
            ORDER BY conv_group)
        FROM UNNEST(c)
    ) AS agg_sessions
FROM agg_by_conv_group