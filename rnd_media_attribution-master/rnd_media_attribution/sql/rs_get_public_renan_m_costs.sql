WITH raw_costs_data AS (
SELECT 
id_cost, id_date, id_device_platform,id_currency, utm_hash,
legal_entity_name, category_name, account_name, supplier_name, short_description,
is_tool, cost_allocation, original_value
FROM public.d20201030_renan_rnd_m_costs
WHERE TRUE  
AND id_store=352
)
SELECT *
FROM raw_costs_data