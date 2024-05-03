SELECT * 
FROM business_layer.fact_marketing_costs fmc
INNER JOIN business_layer.dim_utm du 
ON FMC.fk_utm = du.id 
WHERE TRUE AND fk_date >20200901 AND fk_date <20201001 AND fk_store =352
AND utm_hash= '41faa25e1960a0374d265b643c1c08c1'



--fa7c25efe4aa5184a58018a9d15dd63f
-- 4a30bab97b60969e27c1f567f8680c22
--41faa25e1960a0374d265b643c1c08c1
WITH utm_costs AS (
SELECT DISTINCT fk_utm, COUNT(*) AS cnt,
AVG(mkt_cost) AS avg_mkt_cost,
AVG(original_mkt_cost) AS avg_original_mkt_cost,
AVG(target_mkt_cost) AS avg_target_mkt_cost--, fk_src_cost 
FROM business_layer.fact_marketing_costs fmc
WHERE TRUE AND fk_date >20200901 AND fk_date <20201001 AND fk_store =352
group by fk_utm
)
SELECT * 
FROM  business_layer.dim_utm du 
INNER JOIN utm_costs 
ON utm_costs.fk_utm = du.id 
--WHERE utm_hash= 'ca8c74148f79e82c2418678b20a3b3b5'
ORDER BY avg_mkt_cost DESC

SELECT * 
FROM  business_layer.dim_utm du 
WHERE utm_hash ='41faa25e1960a0374d265b643c1c08c1'
--LIMIT 10


SELECT * 
FROM  integration_layer.int_marketing_costs imc 
WHERE TRUE
AND id_date >= 20200901
AND id_date <= 20201001
AND id_company = 1
AND utm_hash='65e3f209e31fea69fc17d24c00f6ef6d'
--utm_hash ='b88429773d4a0d05c8454d25386b404d'
ORDER BY id_date
LIMIT 10;

SELECT *
FROM aggregated_tables.new_marketing_costs fc
INNER JOIN business_layer.dim_utm du 
        ON du.id_store = fc.id_store
       AND du.id_company=fc.id_company
       AND du.id_channel_partner = fc.id_channel_partner
WHERE TRUE
AND fk_date >= 20200901
AND fk_date <= 20201001
AND fc.id_company = 1
--AND utm_hash ='b88429773d4a0d05c8454d25386b404d'
--AND utm_hash='41faa25e1960a0374d265b643c1c08c1'
AND utm_hash='65e3f209e31fea69fc17d24c00f6ef6d'
ORDER BY fk_date;
;
--GROUP BY local_mkt_cost_total;

WITH AUX AS (
SELECT utm_hash,fk_date, fc.local_mkt_cost_total--, imc.original_value
FROM aggregated_tables.new_marketing_costs fc
INNER JOIN business_layer.dim_utm du 
        ON du.id_store = fc.id_store
       AND du.id_company=fc.id_company
       AND du.id_channel_partner = fc.id_channel_partner
WHERE TRUE
AND fk_date > 20200901
AND fk_date < 20201001
AND fc.id_company = 1
--AND utm_hash ='b88429773d4a0d05c8454d25386b404d'
--AND utm_hash='41faa25e1960a0374d265b643c1c08c1'
AND utm_hash='65e3f209e31fea69fc17d24c00f6ef6d'
)
SELECT AUX.fk_date, AUX.utm_hash, AUX.local_mkt_cost_total, imc.original_value
FROM AUX
INNER JOIN integration_layer.int_marketing_costs imc
		ON imc.utm_hash = AUX.utm_hash
		AND imc.original_value = AUX.local_mkt_cost_total


--SELECT *--fc.utm_hash, fc.local_mkt_cost_total--, imc.original_value
--FROM aggregated_tables.new_marketing_costs fc
--INNER JOIN integration_layer.int_marketing_costs imc 
--		ON imc.utm_hash = fc.utm_hash
--INNER JOIN business_layer.dim_utm du 
--        ON du.id_store = fc.id_store
--       AND du.id_company=fc.id_company
--       AND du.id_channel_partner = fc.id_channel_partner
--WHERE TRUE
--AND fc.fk_date > 20200901
--AND fc.fk_date < 20201001
--AND fc.id_company = 1
----AND utm_hash ='b88429773d4a0d05c8454d25386b404d'
----AND utm_hash='41faa25e1960a0374d265b643c1c08c1'
--AND utm_hash='65e3f209e31fea69fc17d24c00f6ef6d'
--LIMIT 10

-- TESTING QUERIES
with test as (
SELECT 
supplier_name, account_name, short_description,
UPPER(supplier_name) as up_sn, UPPER(account_name) as up_an, UPPER(short_description) as up_sd
FROM public.d20201030_renan_rnd_m_costs2 drrmc
WHERE TRUE 
AND original_value = 45.77
)
SELECT *,
UPPER(LEFT(up_sn,1))+LOWER(SUBSTRING(up_sn,2,LEN(up_sn))) as supplier_altered,
UPPER(LEFT(up_sd,1))+LOWER(SUBSTRING(up_sd,2,LEN(up_sd))) as sd_altered
FROM test

SELECT * FROM business_layer.dim_utm 
WHERE utm_hash = 'aa40b43ff039e5938edebc16b4217436'

SELECT *,
MD5(
    COALESCE((supplier_name), '_')
    || COALESCE((account_name), '_')
    || COALESCE((short_description), '_')
    || COALESCE((channel_name), '_')
    || COALESCE((id_device_platform::TEXT), '_')) as utm_hash0,
MD5(
    COALESCE(UPPER(supplier_name), '_')
    || COALESCE(UPPER(account_name), '_')
    || COALESCE(UPPER(short_description), '_')
    || COALESCE(UPPER(channel_name), '_')
    || COALESCE(UPPER(id_device_platform::TEXT), '_')) as utm_hash2,
MD5(
	COALESCE(LOWER(supplier_name), '_')
    || COALESCE(LOWER(account_name), '_')
    || COALESCE(LOWER(short_description), '_')
    || COALESCE(LOWER(channel_name), '_')
    || COALESCE(LOWER(id_device_platform::TEXT), '_')) as utm_hash3    
FROM public.d20201030_renan_rnd_m_costs2 drrmc
WHERE TRUE 
AND original_value = 45.77



SELECT *
--,MD5(
--    COALESCE(nmc.supplier_name, '_')
--    || COALESCE(nmc.gl_account_name, '_')
--    || COALESCE(nmc.short_description, '_')
--    || COALESCE(nmc.channel_name_madruga, '_')
--    || COALESCE(ddp.pk_device_platform::TEXT, '_')) as utm_hash2,
--MD5(
--	COALESCE(UPPER(nmc.supplier_name), '_')
--	|| COALESCE(UPPER(nmc.gl_account_name), '_')
--	|| COALESCE(UPPER(nmc.short_description), '_')
--	|| COALESCE(UPPER(nmc.channel_name_madruga), '_')
--	|| COALESCE(UPPER(ddp.pk_device_platform::TEXT), '_')) as utm_hash3
FROM aggregated_tables.new_marketing_costs nmc
--INNER JOIN business_layer.dim_device_platform ddp 
--        ON ddp.device_group_3 = nmc.device_group_3 
--INNER JOIN business_layer.dim_utm du 
--        ON du.id_store = nmc.id_store
--       AND du.id_company=nmc.id_company
--       AND du.id_channel_partner = nmc.id_channel_partner
WHERE TRUE 
AND nmc.fk_date = 20200901
--AND nmc.local_mkt_cost_total = 45.77
--AND utm_hash = 'aa40b43ff039e5938edebc16b4217436'
--OR utm_hash = '4d9879b3776b84bd1698c6e214e29c67'
LIMIT 10

SELECT *
FROM business_layer.dim_customer dc 
--WHERE id_hash ='01664aa11c041105dd9f03559efb4429'
WHERE first_name = 'Ramon' AND last_name = 'Rosa'
LIMIT 10

SELECT fc.supplier_name, fc.gl_account_name, fc.short_description, fc.channel_name_madruga,
ddp.pk_device_platform, fc.device_group_3, fc.local_mkt_cost_total,
MD5(
    COALESCE(fc.supplier_name, '_')
    || COALESCE(fc.gl_account_name, '_')
    || COALESCE(fc.short_description, '_')
    || COALESCE(fc.channel_name_madruga, '_')
    || COALESCE(ddp.pk_device_platform::TEXT, '_'))
FROM aggregated_tables.new_marketing_costs fc
INNER JOIN business_layer.dim_device_platform ddp 
        ON ddp.device_group_3 = fc.device_group_3 
--       AND du.id_company=fc.id_company
--       AND du.id_channel_partner = fc.id_channel_partner
WHERE TRUE
AND fc.fk_date = 20200901
--AND fc.fk_date <= 20201001
AND fc.id_company = 1
--AND local_mkt_cost_total = 45.77
--AND utm_hash ='b88429773d4a0d05c8454d25386b404d'
--AND utm_hash='41faa25e1960a0374d265b643c1c08c1'
--AND utm_hash='65e3f209e31fea69fc17d24c00f6ef6d'
ORDER BY fk_date


SELECT * 
FROM  integration_layer.int_marketing_costs imc 
WHERE TRUE
AND id_date == 20200901
AND id_date <= 20201001
AND id_company = 1
AND original_value = 45.77
--AND utm_hash='65e3f209e31fea69fc17d24c00f6ef6d'
--AND utm_hash='aa40b43ff039e5938edebc16b4217436' 
--utm_hash ='b88429773d4a0d05c8454d25386b404d'
ORDER BY id_date
--LIMIT 10;