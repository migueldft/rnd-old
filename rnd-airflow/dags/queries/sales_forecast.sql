SELECT
    to_date(fs.fk_sale_order_store_date, 'YYYYMMDD') AS date_trunc,
    CAST(DATE_PART(w, date_trunc) AS INT) as weekofyear,
    CAST(DATE_PART(weekday, date_trunc) AS INT) as dayofweek,
    CAST(EXTRACT('day' from date_trunc('week', date_trunc) - date_trunc('week', date_trunc('month', date_trunc))) / 7 + 1 AS INT) AS weekofmonth,
    CAST(DATE_PART(y, date_trunc) AS INT) AS year,
    SUM(1) AS sales,
    dpc.sku_config AS sku_config,
    AVG(fs.gross_merchandise_value_no_discount_bef_cnc_bef_ret) AS avg_price,
    MAX(dpc.cmc_category_bp) AS cmc_category_bp,
    MAX(dpc.cmc_business_unit_bp) AS cmc_business_unit_bp,
    MAX(dpc.cmc_division_bp) AS division_bp
FROM business_layer.fact_sales as fs
    LEFT JOIN business_layer.dim_product_config AS dpc ON fs.fk_product_config_store = dpc.id_product_config
WHERE (fs.fk_country=1) AND (fs.sale_order_item_qty_aft_cnc_aft_ret != 0) AND (date_trunc > '2015-01-01')
    GROUP BY date_trunc, sku_config
    ORDER BY date_trunc;
