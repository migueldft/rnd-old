SELECT DISTINCT (id_hash), country_region, gender, DATEDIFF(hour,birthday,GETDATE())/8766 AS trunc_age,
DATEDIFF(hour,customer_created_at,GETDATE())/8766 AS cust_created_years, num_orders
FROM( 
	SELECT fk_customer, fk_address_shipping,	COUNT(1) AS num_orders
	FROM business_layer.fact_sales fs2
	WHERE is_paid=1	AND fk_store=352 AND fk_sale_order_store_date>20200101
	GROUP BY fk_customer, fk_address_shipping 
	) as fs2
JOIN business_layer.dim_customer dc ON fs2.fk_customer = dc.pk_customer
JOIN business_layer.dim_address da ON fs2.fk_address_shipping = da.pk_address
WHERE TRUE AND LENGTH(id_hash)>1