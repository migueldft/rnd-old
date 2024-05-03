drop table if exists #aux;
create table #aux as
        SELECT
            ct.id,
            TO_CHAR(ct.date_mgr::DATE,'YYYYMMDD') AS id_date,
            CASE
                WHEN st.country_id = 9 THEN 0
                ELSE COALESCE(st.country_id, 0)
            END AS id_country,
            CASE
                WHEN ct.store_id = 1 THEN 1
                WHEN ct.store_id = 3 THEN 2
                WHEN ct.store_id = 5 THEN 3
                WHEN ct.store_id = 7 THEN 4
                ELSE 1
            END AS id_company,
            CASE
                WHEN ct.store_id = 1 THEN 352
                WHEN ct.store_id = 3 THEN 219
                WHEN ct.store_id = 5 THEN 21
                WHEN ct.store_id = 7 THEN 50
                WHEN ct.store_id = 13 THEN 158
                WHEN ct.store_id = 14 THEN 4
                WHEN ct.store_id = 11 THEN 66
                WHEN ct.store_id = 15 THEN 67
                WHEN ct.store_id = 16 THEN 30
                WHEN ct.store_id = 17 THEN 18
                WHEN ct.store_id = 19 THEN 37
                WHEN ct.store_id = 20 THEN 70
                WHEN ct.store_id = 21 THEN 24
                WHEN ct.store_id = 23 THEN 47
                WHEN ct.store_id = 29 THEN 55
                WHEN ct.store_id = 26 THEN 92
                WHEN ct.store_id = 25 THEN 6
                WHEN ct.store_id = 24 THEN 888
                ELSE 0
            END AS id_store,
            CASE
                WHEN ct.device_id = 3 THEN 146
                WHEN ct.device_id = 4 THEN 330
                WHEN ct.device_id = 5 THEN 606
                WHEN ct.device_id = 6 THEN 514
                WHEN ct.device_id = 7 THEN 422
                WHEN ct.device_id = 8 THEN 698
                WHEN ct.device_id = 9 THEN 790
                WHEN ct.device_id = 10 THEN 1250
                WHEN ct.device_id = 11 THEN 1342
                WHEN ct.device_id = 13 THEN 238
                ELSE 0
            END AS id_device_platform,
            CASE 
                WHEN ct.currency_id = 1 THEN 173
                WHEN ct.currency_id = 2 THEN 265
                WHEN ct.currency_id = 3 THEN 357
                WHEN ct.currency_id = 4 THEN 449
                WHEN ct.currency_id = 5 THEN 817
                WHEN ct.currency_id = 6 THEN 633
                WHEN ct.currency_id = 7 THEN 541
                WHEN ct.currency_id = 8 THEN 725
                ELSE 0
            END AS id_currency,
            co.legal_entity_name,
            CASE
                WHEN ca.category_name LIKE 'Canais%' THEN 'PERFORMANCE'
                WHEN ca.category_name LIKE 'Strategic Marketing%' THEN 'STRATEGIC MARKETING'
                WHEN ca.category_name LIKE 'Creative Marketing%' THEN 'CREATIVE MARKETING'
                WHEN ca.category_name LIKE 'Trade Marketing%' THEN 'TRADE MARKETING'
                WHEN ca.category_name = 'B2B Services' THEN 'B2B'
                ELSE ca.category_name
            END AS category_name,
            gl.name AS account_name,
            sp.supplier_name,
            mc.channel_name,
            ct.short_description,
            CASE
                WHEN gl.id = 6 THEN 1
                ELSE 0
            END AS is_tool,
            ct.cost_allocation,
            ct.value
        FROM 
            raw_madruga.costs ct
        INNER JOIN
            raw_madruga.stores st 
                ON 
                ct.store_id = st.id
        INNER JOIN
            raw_madruga.suppliers sp 
                ON 
                ct.supplier_id = sp.id
        INNER JOIN
            raw_madruga.categories ca 
                ON 
                ct.category_id = ca.id
        INNER JOIN
            raw_madruga.companies co
                ON 
                ct.company_id = co.id
        INNER JOIN
            raw_madruga.gl_accounts gl 
                ON 
                ct.gl_account_id = gl.id
        INNER JOIN
            raw_madruga.channels mc
                ON
                mc.id = ct.marketing_channel_id
        WHERE 
            ct.is_deleted = 0
            AND
--			partition_field BETWEEN '${START_DATE}' AND '${END_DATE}' ---------------------FALTA PARTITION FIELD NA RAW_MADRUGA.COSTS
            TO_CHAR(ct.date_mgr::DATE, 'YYYYMMDD') BETWEEN '20200901' AND '20201001'
        GROUP BY
            1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
    DROP TABLE IF EXISTS public.d20201030_renan_rnd_m_costs;
    CREATE TABLE public.d20201030_renan_rnd_m_costs as
        SELECT DISTINCT
            t.id as id_cost,
            CASE
                WHEN t.cost_allocation = 1 THEN dd.pk_date
                WHEN t.cost_allocation = 30 THEN d.pk_date
                ELSE dd.pk_date
            END AS id_date,
            t.id_country,
            t.id_company,
            t.id_store,
            t.id_currency,
            MD5(
                COALESCE(t.supplier_name, '_')
                || COALESCE(t.account_name, '_')
                || COALESCE(t.short_description, '_')
                || COALESCE(t.channel_name, '_')
                || COALESCE(t.id_device_platform::TEXT, '_')
            ) AS utm_hash,
            t.legal_entity_name,
            t.category_name,
            t.supplier_name,
            t.account_name,
            t.short_description,
            t.channel_name,
			t.id_device_platform,            
            t.is_tool,
            CASE
                WHEN t.cost_allocation = 1 THEN 'DAILY'
                ELSE 'MONTHLY'
            END AS cost_allocation,
            CASE
                WHEN t.cost_allocation = 1 THEN t.value
                ELSE t.value / MAX(CAST(d.day_of_month AS INT)) OVER (PARTITION BY d.month_of_year)
            END
            * 
            CASE
                WHEN t.id_company = 1 THEN COALESCE(CAST(xrt.brl_rate AS REAL), 1.0)
                WHEN t.id_company = 2 THEN COALESCE(CAST(xrt.ars_rate AS REAL), 1.0)
                WHEN t.id_company = 3 THEN COALESCE(CAST(xrt.clp_rate AS REAL), 1.0)
                WHEN t.id_company = 4 THEN COALESCE(CAST(xrt.cop_rate AS REAL), 1.0)
            ELSE 1
            END AS exchanged_value,
            CASE
                WHEN t.cost_allocation = 1 THEN t.value
                ELSE t.value / MAX(CAST(d.day_of_month AS INT)) OVER (PARTITION BY d.month_of_year)
            END as original_value
        FROM
            #aux t
        INNER JOIN
            business_layer.dim_date d
                ON 
                d.year_month = TO_CHAR(t.id_date::DATE, 'YYYYMM')
        INNER JOIN
            business_layer.dim_date dd
                ON 
                dd.pk_date = t.id_date
        LEFT JOIN
            business_layer.fact_exchange_rate xrt
                ON 
                t.id_currency = xrt.fk_currency
                AND 
                xrt.fk_date = t.id_date;
