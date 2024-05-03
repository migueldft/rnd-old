WITH fmc AS (
	SELECT 
		etl_load_date,
		fk_date,
		fk_device_platform,
		fk_store, 
		fk_campaign_type,
		fk_channel_partner,
		mkt_cost,
		target_mkt_cost 
	FROM business_layer.fact_marketing_costs fmc 
	WHERE fk_date >20200801
)
SELECT 
		source_name, 
		partner_name, 
		channel_name,
		channel_type, 
		partner_group,
		partner_name,
		fk_date,
		fk_device_platform,
		fk_store, 
		fk_campaign_type,
		mkt_cost,
		target_mkt_cost 
	FROM business_layer.dim_channel_partner dcp 
	JOIN fmc on fmc.fk_channel_partner = pk_channel_partner
order by mkt_cost DESC
LIMIT 10