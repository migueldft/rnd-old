WITH raw_customer as (
	SELECT DISTINCT 
		pk_customer,
		src_fk_customer,
		id_hash,
		gender,
		EXTRACT(YEAR FROM birthday) AS birthday,
		EXTRACT(YEAR FROM customer_created_at) AS cust_created_year
	FROM business_layer.dim_customer dc
		WHERE fk_company = 1
)
, billing_address as (
	SELECT DISTINCT 
		fk_customer,
		LAST_VALUE(fk_address_billing)
			OVER(PARTITION BY fk_customer 
				ORDER BY sale_order_store_date 
				ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
			) as fk_address_billing
	FROM business_layer.fact_sales fs
		WHERE fk_company = 1
		AND is_canceled = 0
)
, address as (
	SELECT 
		pk_address,
		country_region 
	FROM business_layer.dim_address da 
)
SELECT  
	src_fk_customer,
	id_hash,
	gender,
	birthday,
	cust_created_year,
	COALESCE(LOWER(country_region),'not_set') as country_region
FROM raw_customer as rc
	LEFT JOIN billing_address as ba 
		ON rc.pk_customer = ba.fk_customer
	LEFT JOIN address as ad
		ON ba.fk_address_billing = ad.pk_address
