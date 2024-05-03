SELECT 
    client_id,
    agg_sessions,
    COALESCE(country_region,'not_set') AS country_region,
    COALESCE(gender,'not_set') AS gender,
    COALESCE(CAST(trunc_age AS STRING),'not_set') AS age,
    COALESCE(CAST(cust_created_years AS STRING),'not_set') AS cust_created_years,
    COALESCE(num_orders,-1) AS num_orders
FROM `dafiti-analytics.RnD_Renan.M_A_user_views_path_toconversion_without_convgroup` AS upath 
FULL OUTER JOIN `dafiti-analytics.RnD_Renan.num_orders_by_idhash` AS num_orders_by_id_hash
          ON upath.client_id = num_orders_by_id_hash.id_hash
WHERE TRUE
    AND LENGTH(agg_sessions)>1
    AND LENGTH(client_id)>1