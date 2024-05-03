WITH raw_medium_data AS(
    SELECT 
        fullvisitorid AS fvid,
        (SELECT value FROM UNNEST(customDimensions) WHERE index=8 AND LENGTH(value)=32) AS idhash,
        visitstarttime AS sess,
        REGEXP_REPLACE(trafficsource.medium, r'[,;:]', '') AS medium,
        COALESCE(totals.totalTransactionRevenue/1e6, 0) AS value,
        IF((SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='6')=0, 0, 1) AS is_conversion
    FROM `dafiti-analytics.40663402.ga_sessions_*`
    WHERE {table_suffix_filter}
), aggregate_by_fvid AS (
    SELECT
        fvid,
        ARRAY_AGG(STRUCT(
            idhash,
            sess,
            medium,
            is_conversion,
            IF(is_conversion=1, value, NULL) AS value
        ) ORDER BY sess) AS s
    FROM raw_medium_data
    GROUP BY fvid
), create_client_id AS (
    SELECT
        ARRAY(
            SELECT AS STRUCT
                * EXCEPT(idhash),
                COALESCE(
                    FIRST_VALUE(idhash IGNORE NULLS) OVER (ORDER BY sess ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING), # next non-null idhash
                    LAST_VALUE(idhash IGNORE NULLS) OVER (ORDER BY sess ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), # previous non-null idhash
                    CONCAT('v',fvid) # if there is no hash id, use fullVisitorId
                ) AS client_id
            FROM UNNEST(s)
        ) AS u
    FROM aggregate_by_fvid 
), get_conversion_group AS (
    SELECT
        u.client_id,
        u.sess,
        u.medium,
        u.value,
        COALESCE(SUM(u.is_conversion) OVER (PARTITION BY u.client_id ORDER BY u.sess ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS conv_group
    FROM create_client_id, UNNEST(u) AS u
), agg_by_client AS (
    SELECT
        client_id,
        ARRAY_AGG(STRUCT(conv_group,sess,medium,value) ORDER BY sess) AS c
    FROM get_conversion_group
    GROUP BY client_id
), agg_by_conv_group AS (
    SELECT
        client_id,
        ARRAY(
            SELECT AS STRUCT
                conv_group,
                STRING_AGG(medium,',' ORDER BY sess) AS media,
                SUM(value) AS value
            FROM UNNEST(c)
            GROUP BY conv_group
        ) AS c
        FROM agg_by_client
) SELECT
    client_id,
    (
        SELECT
            STRING_AGG( CONCAT(media, ':', CAST(COALESCE(value,-1) AS STRING)) , ';'
            ORDER BY conv_group)
        FROM UNNEST(c)
    ) AS agg_sessions
FROM agg_by_conv_group