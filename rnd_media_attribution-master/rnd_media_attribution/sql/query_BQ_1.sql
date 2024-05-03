SELECT
  fullvisitorid,
  visitstarttime,
  COALESCE(totals.bounces, 0) AS is_bounce,
  trafficsource.medium,
  ARRAY(
    SELECT AS STRUCT
      h.ecommerceaction.action_type AS act,
      COUNT(1) AS cnt
    FROM UNNEST(hits) AS h
    GROUP BY act
  ) as struct_action,
  ARRAY(
    SELECT AS STRUCT
      --hp.productSKU AS sku,
      hp.isimpression as is_impression,
      (CASE
        WHEN h.ecommerceaction.action_type='1' THEN 'prod_detail_views'
        WHEN h.ecommerceaction.action_type='2' THEN 'add_to_cart'
        WHEN h.ecommerceaction.action_type='3' THEN 'remove_from_cart'
        WHEN h.ecommerceaction.action_type='4' THEN 'check_out'
        WHEN h.ecommerceaction.action_type='5' THEN 'complete_purchase'
        WHEN h.ecommerceaction.action_type='6' THEN 'refund_purchase'
        WHEN h.ecommerceaction.action_type='7' THEN 'refund_purchase'
        WHEN h.ecommerceaction.action_type='8' THEN 'check_out_options'
        WHEN h.ecommerceaction.action_type='0' THEN 'unkown'
        ELSE NULL
      END) AS act_name,
    FROM UNNEST(hits) AS h, UNNEST(h.product) AS hp
    --WHERE hp.isimpression IS NOT TRUE
) as impression
FROM
  `dafiti-analytics.40663402.ga_sample`
--  `dafiti-analytics.40663402.ga_sessions_*`
--WHERE _TABLE_SUFFIX BETWEEN '20200101' AND '20200101'
-- GROUP BY fullvisitorid, visitid, visitstarttime, is_bounce, trafficsource.medium
WHERE fullVisitorId ='1000194599418083176'
ORDER BY fullVisitorId, visitStartTime




