SELECT
    COUNT(1) AS num_products,
    mkt_product_division_1,
    mkt_product_division_2,
    cmc_category
FROM business_layer.dim_product_config
WHERE is_online = 1
GROUP BY
    mkt_product_division_1,
    mkt_product_division_2,
    cmc_category