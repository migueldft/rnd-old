DROP TABLE IF EXISTS public.rnd_blackfriday2020_normalized_share;
CREATE TABLE public.rnd_blackfriday2020_normalized_share AS (
	WITH thanksgiving_dates AS (
		SELECT
			"year" AS y,
			date_ AS d,
			(RANK() OVER (PARTITION BY "year" ORDER BY day_of_month))=4 AS is_fourth_thursday
		FROM business_layer.dim_date
		WHERE TRUE
			AND "year">=2011
			AND month_of_year=11
			AND day_of_week=4
		ORDER BY d
	), blackfriday_dates AS (
		SELECT
			y,
			DATE_ADD('day', 1, d)::DATE AS bf_date
		FROM thanksgiving_dates
		WHERE is_fourth_thursday
	), get_sales AS (
		SELECT
			sale_order_store_date::date,
			EXTRACT(HOUR FROM sale_order_store_date) AS h,
			EXTRACT(year FROM sale_order_store_date) AS y,
			gross_total_value_aft_cnc_bef_ret AS gtv_acbr,
			gross_total_value_bef_cnc_bef_ret AS gtv_bcbr,
			(gross_total_value_bef_cnc_bef_ret <> 0)::int AS p
		FROM business_layer.fact_sales
		WHERE fk_company=1
	), join_bfd AS (
		SELECT
			gs.*,
			bfd.bf_date
		FROM get_sales gs
		LEFT JOIN blackfriday_dates bfd
			ON gs.y = bfd.y
	), groupdays AS (
			SELECT
				y,
				DATEDIFF(DAY, bf_date, sale_order_store_date) AS d,
				h,
				ROUND(SUM(gtv_acbr),2) AS gtv_acbr,
				ROUND(SUM(gtv_bcbr),2) AS gtv_bcbr,
				SUM(p) AS p
			FROM join_bfd
			WHERE TRUE
				AND DATEDIFF(DAY, bf_date, sale_order_store_date) BETWEEN -1 AND 5
				AND y BETWEEN EXTRACT(YEAR FROM CURRENT_DATE)-4 AND EXTRACT(YEAR FROM CURRENT_DATE)-1
			GROUP BY y, d, h
	), normalize_qtties AS (
		SELECT
			y,d,h,
			gtv_acbr/(SUM(gtv_acbr) OVER (PARTITION BY y)) AS norm_gtv_acbr,
			gtv_bcbr/(SUM(gtv_bcbr) OVER (PARTITION BY y)) AS norm_gtv_bcbr,
			p::float/(SUM(p) OVER (PARTITION BY y)) AS norm_purchases
		FROM groupdays
		ORDER BY y, d, h
	), sum_past_years AS (
		SELECT
			d,
			h,
			SUM(norm_gtv_acbr) AS xnorm_gtv_acbr,
			SUM(norm_gtv_bcbr) AS xnorm_gtv_bcbr,
			STDDEV_SAMP(norm_gtv_acbr) AS std_norm_gtv_acbr,
			STDDEV_SAMP(norm_gtv_bcbr) AS std_norm_gtv_bcbr,
			SUM(norm_purchases) AS xnorm_purchases,
			STDDEV_SAMP(norm_purchases) AS std_norm_purchases
		FROM normalize_qtties
		GROUP BY d, h
	)
	SELECT
		DATEADD(
			HOUR,
			24*d + h,
			(SELECT bf_date FROM blackfriday_dates WHERE y=EXTRACT(YEAR FROM CURRENT_DATE) LIMIT 1)
		) AS dtt,
		xnorm_gtv_acbr / (SUM(xnorm_gtv_acbr) OVER ()) AS norm_gtv_acbr,
		std_norm_gtv_acbr,
		xnorm_gtv_bcbr / (SUM(xnorm_gtv_bcbr) OVER ()) AS norm_gtv_bcbr,
		std_norm_gtv_bcbr,
		xnorm_purchases / (SUM(xnorm_purchases) OVER ()) AS norm_purchases,
		std_norm_purchases
	FROM sum_past_years
	ORDER BY dtt
)