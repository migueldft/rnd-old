WITH br_mkt_costs as(
SELECT
	id_date, utm_hash as utm, legal_entity_name, category_name, account_name, supplier_name, short_description, is_tool, cost_allocation,exchanged_value,original_value
FROM public.d20200830_renan_rnd_m_costs
WHERE id_country=1
)
SELECT utm_hash,id_date,origin,campaign, source as src, medium, keyword, adcontent, channel_partner, partner_group_1,partner_group_2,partner_group_3,legal_entity_name, category_name, account_name, supplier_name, short_description, is_tool, cost_allocation,exchanged_value,original_value
FROM br_mkt_costs 
JOIN business_layer.dim_utm du on du.utm_hash = utm




