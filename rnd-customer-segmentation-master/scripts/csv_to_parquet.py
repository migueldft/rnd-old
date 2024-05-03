import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

columns = ['date', 'customer_id', 'n_of_orders', 'customer_gender', 
'customer_age_at_date', 'customer_created_at', 'customer_first_order_paid', 
'channel_partner_name', 'sku_config', 'is_campaign', 'product_name', 
'product_gender', 'product_color', 'product_brand', 'product_medium_image_url', 
'cmc_category_bp', 'cmc_division_bp', 'cmc_business_unit_bp', 'google_product_category', 
'product_original_price', 'payment_method_name', 'shipping_condition', 'product_discount', 
'shipping_discount', 'sale_value', 'is_coupon', 'device_name', 'platform_name', 
'delivery_city', 'delivery_city_state_code', 'delivery_country_region', 
'planning_cluster', 'planning_age', 'ticketrange_planning', 'originalpricerange_planning']

data = pd.read_csv('/home/ubuntu/data/raw/user-clustering-data-17-07-2019.csv', names=columns, sep=';')

pa_table = pa.Table.from_pandas(data)
pq.write_table(pa_table, '/home/ubuntu/data/preprocessed/user-clustering-data-17-07-2019.parquet')