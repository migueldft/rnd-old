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
        geonetwork.region as region,
        (CASE
          WHEN datasource='app' THEN (SELECT value FROM unnest(customDimensions) WHERE index=23)
          ELSE NULL
        END) AS device_id,
        LOWER(REGEXP_REPLACE(trafficsource.source, r'[,;:]', '')) AS src,
        LOWER(REGEXP_REPLACE(trafficsource.medium, r'[,;:]', '')) AS medium,
        (IF(REGEXP_REPLACE(trafficsource.keyword, r'[,;:]', '') LIKE '%dafit%',1,0)) AS is_some_kw_dft,
        IF(totals.totalTransactionRevenue/1e6 > 0,1, 0) AS is_conversion,
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
        * EXCEPT(src, medium, is_some_kw_dft, is_pagedirect),

        IF(is_pagedirect=1 AND src NOT IN ('organic','cpc','(direct)') AND is_some_kw_dft != 1,
            '(direct)',src) AS src,
        IF(is_pagedirect=1 AND src NOT IN ('organic','cpc','(direct)') AND is_some_kw_dft != 1,
            '(none)',medium) AS medium,
        IF(is_pagedirect=1 AND src NOT IN ('organic','cpc','(direct)') AND is_some_kw_dft != 1,
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
    FROM parse_customer_id as pcid
    LEFT JOIN `dafiti-analytics.RnD_Renan.rs_dim_customer` as dc1
      ON pcid.sfc = cast (dc1.src_fk_customer as string)
    LEFT JOIN `dafiti-analytics.RnD_Renan.rs_dim_customer` as dc2
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
        ARRAY_AGG(
            (SELECT AS STRUCT * EXCEPT(gid)
            FROM UNNEST([create_global_id]))
        ORDER BY sess) AS v
    FROM create_global_id
    GROUP BY gid
), create_conv_group AS (
    SELECT 
        gid,
--         c.*,
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
              (MAX(is_conversion) OVER (PARTITION BY conv_group)) as cv_marker,
              CONCAT (src,'_',medium,'_',is_some_kw_dft) as trin
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
), add_start_on_sessions as (
    SELECT
        * EXCEPT(v),
        ARRAY(
              (SELECT AS STRUCT -1 AS sess, 'START' AS trin, 0 AS cv_marker)
              UNION ALL
              (SELECT AS STRUCT v.sess, v.trin, v.cv_marker FROM UNNEST(v) AS v)
            ) AS v
    FROM remove_sequential_sessions
), create_nex_medium as (
    SELECT 
        * except (v),
        ARRAY(
            SELECT AS STRUCT 
                * EXCEPT(o),
                (CASE WHEN cv_marker = 0 THEN
                    COALESCE(LEAD(trin) OVER(PARTITION BY gid ORDER BY o),'END')
                    ELSE COALESCE(LEAD(trin) OVER(PARTITION BY gid ORDER BY o),'CONVERSION')
                END) AS next_medium
            FROM UNNEST(v)
            WITH OFFSET AS o
            ORDER BY o
        ) AS v
    FROM add_start_on_sessions
), unnest_all as (
    SELECT 
        v.* 
    FROM create_nex_medium, unnest(v) as v
)
SELECT
    trin,
    next_medium,
    COUNT(*) as cnt,
FROM unnest_all
GROUP BY trin,next_medium
ORDER BY trin,next_medium