WITH 
    #aux0 AS (
        select 
            isi.fk_customer
            , isi.sale_order_store_date
            , isi.sale_order_store_number
        from integration_layer.int_sale_item isi
            inner join business_layer.dim_store ds on ds.pk_store = isi.fk_store
        where 1
            and isi.is_paid = 1
            and ds.store_group = 'DAFITI BRAZIL'
        group by 1,2,3
    ),

    #aux1 AS (
        select 
            aux.fk_customer
            , aux.first_sale_date
            , aux.first_sale_number
            --, to_char(aux.first_sale_date, 'YYYYMMDD') as date_test
        from
            (select #aux0.fk_customer,
            first_value(#aux0.sale_order_store_date) over (partition by #aux0.fk_customer order by #aux0.sale_order_store_date rows between unbounded preceding and unbounded following) as first_sale_date,
            first_value(#aux0.sale_order_store_number) over (partition by #aux0.fk_customer order by #aux0.sale_order_store_date rows between unbounded preceding and unbounded following) as first_sale_number
        from #aux0) aux
            where aux.first_sale_date >= '2018-01-01'
        group by 1,2,3
    ),
    
    #aux2 AS (
        select
            isi.fk_customer
            , cp.channel_name
            , cp.partner_name
            , dp.device_group_3
            , dc.birthday
            , dc.gender
            , dda.state
            , isi.delivery_date_agreed_with_customer
            , isi.delivered_date
            , isi.sale_order_store_number
            , isi.sale_order_store_date
        from dftdwh.integration_layer.int_sale_item isi
            inner join business_layer.dim_customer dc on dc.pk_customer = isi.fk_customer
            inner join business_layer.dim_delivery_address dda on dda.pk_address = isi.fk_address_shipping
            inner join business_layer.dim_store ds on ds.pk_store = isi.fk_store
            inner join business_layer.dim_channel_partner cp on cp.pk_channel_partner = isi.fk_last_click_channel_partner
            inner join business_layer.dim_device_platform dp on dp.pk_device_platform = isi.fk_device_platform
        where 1 
            and isi.fk_customer in (select fk_customer from #aux1)
            and isi.is_paid = 1
            and ds.store_group = 'DAFITI BRAZIL'
        group by 1,2,3,4,5,6,7,8,9,10,11
    ),

    #features AS (
        select 
            db.fk_customer
            , min(db.channel_name) as channel
            , min(db.partner_name) as partner
            , min(db.device_group_3) as device
            , min(db.first_sale_number) as first_sale_number
            , min((extract(year from sysdate) - extract(year from db.birthday))) as age
            , min(db.gender) as gender
            , min(lower(db.state)) as state
            , to_char(min(db.delivery_date_agreed_with_customer),'YYYYMMDD') as expected_delivery_date
            , to_char(min(db.delivered_date),'YYYYMMDD') as delivered_date
            , to_char(min(db.first_sale_date),'YYYYMMDD') as first_sale_date
            , to_char(min(db.second_sale_date),'YYYYMMDD') as second_sale_date
        from (select 
            #aux2.fk_customer,
            #aux2.sale_order_store_number,
            first_value(#aux2.sale_order_store_date) over (partition by #aux2.fk_customer order by #aux2.sale_order_store_date rows between unbounded preceding and unbounded following) as first_sale_date,
            nth_value(#aux2.sale_order_store_date,2) over (partition by #aux2.fk_customer order by #aux2.sale_order_store_date rows between unbounded preceding and unbounded following) as second_sale_date,
            first_value(#aux2.sale_order_store_number) over (partition by #aux2.fk_customer order by #aux2.sale_order_store_date rows between unbounded preceding and unbounded following) as first_sale_number,
            #aux2.channel_name,
            #aux2.partner_name,
            #aux2.device_group_3,
            #aux2.birthday,
            #aux2.gender,
            #aux2.state,
            #aux2.delivered_date,
            #aux2.delivery_date_agreed_with_customer
        from #aux2 ) db
        group by 1
    ),

    #targets AS (
        select 
            isi.sale_order_store_number as first_sale_number
            , isi.fk_customer
            , case when sum(pc.is_market_place_in) > 0 then 1 else 0 end as has_marketplace
            , case when sum(pc.is_crossdocking) > 0 then 1 else 0 end as has_crossdocking
            , case when sum(pc.is_private_label) > 0 then 1 else 0 end as has_private_label
            , case when sum(case when pc.category_erp in ('Vestuário Brands', 'Calçados Brands', 'Acessórios Brands', 'Vestuário Premium', 'Acessórios Premium', 'Calçados Premium') then 1 else 0 end) > 0 then 1 else 0 end as has_brands
            , sum(isi.gross_merchandise_value) as gmv 
        from integration_layer.int_sale_item isi
            inner join business_layer.dim_store ds on ds.pk_store = isi.fk_store
            inner join business_layer.dim_product_config pc on isi.fk_product_config_store = pc.id_product_config
            inner join #aux1 on #aux1.first_sale_number = isi.sale_order_store_number
        where 1
            --and isi.sale_order_store_number in(select DISTINCT(sale_order_store_number) from #aux1)
            and ds.store_group = 'DAFITI BRAZIL'
        group by 1,2
    )


SELECT
    f.fk_customer
    , f.channel
    , f.partner
    , f.device
    , f.first_sale_number
    , f.age
    , f.gender
    , f.state
    , f.expected_delivery_date
    , f.delivered_date
    , f.first_sale_date
    , f.second_sale_date
    , t.has_marketplace
    , t.has_crossdocking
    , t.has_private_label
    , t.has_brands
    , t.gmv
FROM #features AS f
    INNER JOIN #targets AS t ON t.first_sale_number = f.first_sale_number AND t.fk_customer = f.fk_customer
