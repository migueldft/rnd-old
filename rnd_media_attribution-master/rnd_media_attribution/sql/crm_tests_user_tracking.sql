WITH raw_crm_braze_users as (
SELECT external_user_id,os,is_control,
REGEXP_EXTRACT(external_user_id, r'^(.*?)-.*?$') AS fk_customer,
REGEXP_EXTRACT(external_user_id, r'^.*?-(.*?)$') AS country,
FROM `dafiti-analytics.RnD_Renan.crm_lift_braze_users` 
)
SELECT *
FROM `dafiti-analytics.RnD_Renan.M_A_cross_device_withconvgroup` as ma
  LEFT JOIN raw_crm_braze_users 
    ON ma.gid = raw_crm_braze_users.fk_customer
WHERE external_user_id is not null