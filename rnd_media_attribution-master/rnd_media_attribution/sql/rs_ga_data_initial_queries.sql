SELECT * --id_date, id_customer, id_transaction, utm_hash, device_devicecategory, trafficsource_istruedirect, event, visit_date
FROM integration_layer.int_google_analytics_conversion_path
WHERE TRUE AND id_company=1 AND id_store=4
LIMIT 10

SELECT *
FROM integration_layer.int_gan_sessions_per_channel -- NADA AQUI
LIMIT 10

SELECT *
FROM integration_layer.int_google_analytics_product_funnel
--WHERE TRUE AND checkouts>0 AND id_date<20200701 -- nao tem nada menor que 20200701
LIMIT 10

SELECT *
FROM business_layer.dim_utm
--WHERE TRUE AND checkouts>0 AND id_date<20200701 -- nao tem nada menor que 20200701
LIMIT 10


SELECT *
FROM business_layer.fact_google_analytics_product_performance
LIMIT 10

SELECT *
FROM business_layer.fact_google_analytics_shopping_behavior
LIMIT 10
