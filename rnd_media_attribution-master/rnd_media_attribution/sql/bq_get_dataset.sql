WITH raw_data_app AS (
    SELECT
        'app' AS datasource,
        *
    FROM `dafiti-app.78143321.ga_sessions_*`
    WHERE {table_suffix_filter}

), raw_data_site AS (
    SELECT
        'site' AS datasource,
        *
    FROM `dafiti-analytics.40663402.ga_sessions_*`
    WHERE {table_suffix_filter}
), union_tables AS (
    SELECT * FROM raw_data_app
    UNION ALL
    SELECT * FROM raw_data_site
), select_data AS (
    SELECT
        datasource,
        fullVisitorId AS fvid,
        visitstartTime AS sess,
        visitstartTime + COALESCE(totals.timeOnSite,0) AS sess_end,
        (SELECT value FROM unnest(customDimensions) WHERE index=8) AS cd8,
        (CASE
          WHEN datasource='app' THEN (SELECT value FROM unnest(customDimensions) WHERE index=23)
          ELSE NULL
        END) AS device_id,
        IF(LOWER(geonetwork.country)='brazil', LOWER(geonetwork.region), 'foreign_country') AS state,
        TO_HEX(MD5(CONCAT(
              COALESCE(trafficSource.campaign, '_'), 
              COALESCE(trafficSource.source, '_'),
              COALESCE(trafficSource.medium, '_'), 
              COALESCE(trafficSource.keyword, '_'), 
              COALESCE(trafficSource.adContent, '_')
        ))) AS utm_hash,
        LOWER(REGEXP_REPLACE(trafficsource.source, r'[,;:]', '')) AS src,
        LOWER(REGEXP_REPLACE(trafficsource.medium, r'[,;:]', '')) AS medium,
        (IF(REGEXP_REPLACE(trafficsource.keyword, r'[,;:]', '') LIKE '%dafit%',1,0)) AS is_some_kw_dft,
        IF((SELECT COUNT(1) FROM UNNEST(hits) as h WHERE h.ecommerceaction.action_type='6')=0, 0, 1) AS is_conversion,
        IF(trafficsource.isTrueDirect,1,0) as istd,
        (CASE 
          WHEN datasource = 'site' THEN 
            IF(
              (SELECT page.pagePath FROM UNNEST(hits) WITH offset as o ORDER BY o LIMIT 1)
              ='/',
              1,0)
            ELSE NULL
        END) AS is_pagedirect,
    FROM union_tables
), is_direct_pagepath_manual_rules AS (
    SELECT 
        * EXCEPT(utm_hash, src, medium, is_some_kw_dft, is_pagedirect),
        
        IF(is_pagedirect=1 AND istd=1 AND is_some_kw_dft != 1,
            '42b4c899c39ac03cc595b36e8a618951', utm_hash) as utm_hash,
        IF(is_pagedirect=1 AND istd=1 AND is_some_kw_dft != 1,
            '(direct)',src) AS src,
        IF(is_pagedirect=1 AND istd=1 AND is_some_kw_dft != 1,
            '(none)',medium) AS medium,
        IF(is_pagedirect=1 AND istd=1 AND is_some_kw_dft != 1,
            1, is_some_kw_dft) AS is_some_kw_dft,
    FROM select_data
), parse_customer_id AS (
    SELECT
        * EXCEPT(cd8, device_id),
        REGEXP_EXTRACT(device_id, r'^[0-9a-fA-F\-]{36}$') AS device_id,
        REGEXP_EXTRACT(cd8, r'^[0-9a-fA-F]{32}$') AS hash_id,
        REGEXP_EXTRACT(cd8, r'^[0-9]+$') AS sfc,
    FROM is_direct_pagepath_manual_rules
), join_with_dim_customer AS (
    SELECT
        pcid.* EXCEPT(hash_id, sfc),
        COALESCE(pcid.hash_id, dc1.id_hash) as hash_id,
        COALESCE(pcid.sfc, CAST (dc2.src_fk_customer AS STRING)) as sfc,
        STRUCT(
            COALESCE (dc1.gender,dc2.gender,'not_set') as gender,
            COALESCE (
                EXTRACT(YEAR FROM CAST(dc1.birthday AS DATETIME)),
                EXTRACT(YEAR FROM CAST(dc2.birthday AS DATETIME)),
                -1) as birthday,
            COALESCE (
                EXTRACT(YEAR FROM CAST(dc1.customer_created_at AS DATETIME)),
                EXTRACT(YEAR FROM CAST(dc2.customer_created_at AS DATETIME)),
                -1) as cust_created_year
        ) AS c
    FROM parse_customer_id as pcid
    LEFT JOIN `dafiti-analytics.curated_data.behaviour_tracking_customer` as dc1
      ON pcid.sfc = cast (dc1.src_fk_customer as string)
    LEFT JOIN `dafiti-analytics.curated_data.behaviour_tracking_customer` as dc2
      ON pcid.hash_id = dc2.id_hash
), get_id_from_other_session AS (
    SELECT
        *,
        ANY_VALUE(COALESCE(sfc, hash_id)) OVER (PARTITION BY COALESCE(device_id, CONCAT(datasource, ':', fvid))) AS other_session_id        
    FROM join_with_dim_customer 
), create_global_id AS (
    SELECT
        COALESCE(
            sfc,
            hash_id,
            other_session_id,
            device_id,
            CONCAT(datasource, ':', fvid)
        ) AS gid,
        * EXCEPT(fvid, device_id, sfc, hash_id, other_session_id, datasource)
        ,
    FROM get_id_from_other_session
), nest_sessions AS (
    SELECT
        gid, 
        any_value(c) as c, 
        ARRAY_AGG(
            (SELECT AS STRUCT * EXCEPT(gid,c),
            FROM UNNEST([create_global_id]))
        ORDER BY sess) AS v
    FROM create_global_id
    GROUP BY gid
), create_conv_group AS (
    SELECT 
        gid,
        c.*,
        ARRAY(SELECT AS STRUCT
            * EXCEPT (off),
            COALESCE(SUM(is_conversion) OVER (PARTITION BY gid ORDER BY sess ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS conv_group
        FROM UNNEST(v) WITH OFFSET AS off ORDER BY off ) AS v
    FROM nest_sessions
), tag_sequential_sessions AS (
    SELECT
        * EXCEPT(v),
        ARRAY(
            SELECT AS STRUCT
                * EXCEPT(o),
                COALESCE(
                    (sess-LAG(sess_end, 1) OVER (ORDER BY o)) > 4*60*60,
                    TRUE
                ) AS is_not_sequence,
              (MAX(is_conversion) OVER (PARTITION BY conv_group)) as cv_marker
            FROM UNNEST(v)
            WITH offset as o
            ORDER BY o
        ) AS v
    FROM create_conv_group
), remove_sequential_sessions AS (
    SELECT
        * EXCEPT(v),
        ARRAY(
            SELECT AS STRUCT
                * EXCEPT(
                is_not_sequence,o),
            FROM UNNEST(v)
            WITH offset as o
            WHERE is_not_sequence
            ORDER BY o
        ) AS v,
    FROM tag_sequential_sessions
), agg_in_line as (
    SELECT 
      * EXCEPT (v),
      ARRAY(
        SELECT AS STRUCT
          conv_group,
          MAX(cv_marker) as cv_marker,
          ANY_VALUE(state) AS state,
          COALESCE(SUM(is_conversion),-1) AS is_conversion,
          STRING_AGG(utm_hash,',' ORDER BY sess) as utm_hash,
          STRING_AGG(CAST (sess AS STRING ),',' ORDER BY sess) as sess,
          STRING_AGG(concat(src,'_',medium,'_',is_some_kw_dft),','ORDER BY sess) as trinomials,
        FROM UNNEST (v)
        group by conv_group
        ) as s
    FROM remove_sequential_sessions as rss
)
SELECT 
    * EXCEPT (s),
    (SELECT COALESCE(ANY_VALUE(state),'not_set') FROM UNNEST (s)) as country_region,
    (SELECT STRING_AGG(CONCAT(conv_group)) FROM UNNEST (s)) as agg_cvg,
    (SELECT STRING_AGG(CONCAT(utm_hash,':',sess,':',trinomials,':', CAST(cv_marker AS STRING)), ';') FROM UNNEST (s)) as agg_sessions
FROM agg_in_line
