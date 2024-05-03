DROP TABLE IF EXISTS #filter_n_orders, #product_clusterization_planning, #product_clusterization_planning_ext, #table_1;

CREATE TABLE #filter_n_orders AS (
SELECT
    dct.pk_customer AS customer_id
    , TO_DATE(isi.fk_sale_order_store_date, 'YYYYMMDD') AS date
    , RANK() OVER (
	        PARTITION BY customer_id
	        ORDER BY TO_DATE(isi.fk_sale_order_store_date, 'YYYYMMDD')
	    ) AS ranking
FROM integration_layer.int_sale_item AS isi
    JOIN business_layer.dim_customer AS dct
        ON dct.pk_customer = isi.fk_customer
    JOIN business_layer.dim_company AS dcp
        ON dcp.pk_company = isi.fk_company_store
WHERE (dcp.company_name = 'DAFITI BRAZIL')
    AND (isi.is_canceled = 0)
    AND (date > '2015-01-01')
GROUP BY customer_id, TO_DATE(isi.fk_sale_order_store_date, 'YYYYMMDD')
);

CREATE TABLE #product_clusterization_planning AS (
SELECT
	cmcbusinessunitbp AS cmc_business_unit_bp
	, cmcdivisionbp AS cmc_division_bp
	, brand
	, cluster
	, age AS age
FROM 
	manual_input.product_clusterization_planning
);

CREATE TABLE #product_clusterization_planning_ext AS (
SELECT
	countrycode
	, cmcbusinessunitbp AS cmc_business_unit_bp
	, cmcdivisionbp AS cmc_division_bp
	, cmccategorybp AS cmc_category_bp
	, brand
	, ticketrange
	, originalpricerange
FROM 
	manual_input.product_clusterization
);

CREATE TABLE #table_1 AS (
SELECT
	TO_DATE(isi.fk_sale_order_store_date, 'YYYYMMDD') AS date
	, dct.pk_customer AS customer_id
	, q.n_orders
	, dct.gender AS customer_gender
	, DATEDIFF(year, dct.birthday, date) AS customer_age
	, TRUNC(dct.customer_created_at) AS customer_created_at
	, TRUNC(dct.date_of_first_order_paid) AS customer_first_order_paid
	, cpt.partner_name AS channel_partner_name
	, dpc.sku_config AS sku_config
	, CASE WHEN isi.fk_campaign != 0 THEN 1 ELSE 0 END AS is_campaign
	, dpc.product_name AS product_name
    , dpc.gender AS product_gender
    , dpc.color AS product_color
    , dpc.brand AS product_brand
    , dpc.product_medium_image AS product_medium_image_url
    , dpc.cmc_category_bp AS cmc_category_bp
    , dpc.cmc_division_bp AS cmc_division_bp
    , dpc.cmc_business_unit_bp AS cmc_business_unit_bp
    , NVL(feed.google_product_category, 'Unknown') AS google_product_category
    , dpc.original_price AS product_original_price
    , dpm.payment_method_name AS payment_method_name
    , ddt.shipping_condition AS shipping_condition
    , isi.product_discount AS product_discount
    , isi.shipping_discount AS shipping_discount
    , isi.gross_total_value AS sale_value
    , CASE WHEN isi.fk_discount_coupon != 0 THEN 1 ELSE 0 END AS is_coupon
    , ddp.device_name AS device_name
    , ddp.platform_name AS platform_name
    , dad.city AS delivery_city
    , dad.state_code AS delivery_state_code
    , dad.country_region AS delivery_country_region
    , CASE WHEN pcp.cluster = ' ' THEN 'Unknown' ELSE pcp.cluster END AS planning_cluster
    , CASE WHEN pcp.age = ' ' THEN 'Unknown' ELSE pcp.age END AS planning_age
    , pcpe.ticketrange AS ticketrange_planning
    , pcpe.originalpricerange AS originalpricerange_planning
FROM integration_layer.int_sale_item AS isi
    JOIN business_layer.dim_customer AS dct
        ON dct.pk_customer = isi.fk_customer
    JOIN business_layer.dim_company AS dcp
        ON dcp.pk_company = isi.fk_company_store
    JOIN business_layer.dim_channel_partner AS cpt
        ON cpt.pk_channel_partner = isi.fk_channel_partner
    JOIN business_layer.dim_product_config AS dpc
        ON dpc.id_product_config = isi.fk_product_config_store
    JOIN business_layer.dim_payment_method AS dpm
        ON dpm.pk_payment_method = isi.fk_payment_method
    JOIN business_layer.dim_device_platform AS ddp
        ON ddp.pk_device_platform = isi.fk_device_platform
    JOIN business_layer.dim_delivery_type AS ddt
        ON ddt.pk_delivery_type = isi.fk_delivery_type
    JOIN business_layer.dim_address AS dad
        ON dad.pk_address = isi.fk_delivery_address
    JOIN #product_clusterization_planning AS pcp
    	ON (
    	pcp.cmc_business_unit_bp = dpc.cmc_business_unit_bp
    	AND 
    	pcp.cmc_division_bp = dpc.cmc_division_bp
    	AND
    	pcp.brand = dpc.brand)
    JOIN #product_clusterization_planning_ext AS pcpe
    	ON (
    	pcpe.cmc_business_unit_bp = dpc.cmc_business_unit_bp
    	AND
    	pcpe.cmc_division_bp = dpc.cmc_division_bp
    	AND
    	pcpe.cmc_category_bp = dpc.cmc_category_bp
    	AND
    	pcpe.brand = dpc.brand)
    LEFT JOIN feed.xml_config_base_dafiti_br AS feed
        ON feed.sku_config = dpc.sku_config
	LEFT JOIN (
		SELECT
			q.customer_id,
			MAX(q.ranking) as n_orders
		FROM #filter_n_orders AS q
		GROUP BY q.customer_id
	) AS q
		ON dct.pk_customer = q.customer_id
WHERE (dcp.company_name = 'DAFITI BRAZIL')
    AND (isi.is_canceled = 0)
    AND (q.n_orders > 1)
    AND (date > '2015-01-01')
);