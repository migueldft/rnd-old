--DROP TABLE IF EXISTS public.d20200830_renan_rnd_sess_inform;
--CREATE TABLE public.d20200830_renan_rnd_sess_inform as (
--SELECT partition_field, partition_value, clientid, userid, fullvisitorid, visitstarttime,visitid, utm_hash,totals_hits, totals_transactions 
--FROM spc_raw_google_analytics_app_dafiti_br.sessions
--WHERE partition_field >20200823
--)

with TEST AS (
SELECT clientid,
	FLOOR(LENGTH(listagg (visitid,','))/10) AS sess_len, -- gambiarra kkkk
	listagg (userid,',')as userid,
	listagg (fullvisitorid,',')as fvid,
	listagg (utm_hash,',')as utm_hash,
--	listagg (visitid,',')as visitid, -- -> same as visitstarttime
	listagg (visitstarttime,',') as visitstarttime,
	listagg (COALESCE (totals_hits,-1),',')as totals_hits,
	listagg (COALESCE (totals_transactions,-1),',') within group (order by visitstarttime) as totals_transactions 
FROM public.d20200830_renan_rnd_sess_inform
--WHERE COALESCE (clientid,'-1')='-1' -- no results -> no clientid is null
--WHERE COALESCE (userid,'-1')!='-1' -- no results-> every userid is null
GROUP BY clientid
)
SELECT *
FROM TEST
--WHERE LENGTH(totals_transactions)>2 --AND clientid ='0000c421-d0db-4320-9af2-7dfa124ada18'--clientid ='000145a4-5424-488d-995c-7703e1fd480b'
LIMIT 10



-- from adjust - not working


SELECT
	user_id,click_time,
	listagg(md5(
				coalesce(campaign_name, '')
				|| coalesce(network_name, '')
				|| coalesce(adgroup_name, '')
				|| coalesce(creative_name, ''))
			, ',') as utm_hash,
--   listagg(lower(campaign_name),',') AS campaign,
--   listagg(lower(network_name),',') AS source,
--   NULL AS medium,
--   listagg(lower(adgroup_name),',') AS keyword,
--   listagg(lower(creative_name),',') AS adcontent,
--	listagg(click_time,',') as sess,
	listagg(COALESCE(value,-1),',') within group (order by click_time) as value
FROM
	spc_raw_adjust.events_dafiti 
WHERE
	partition_field BETWEEN '20200901' AND '20200901'
	AND country = 'br'
	AND user_id IS NOT NULL
	AND user_id = 6553486
GROUP BY user_id, click_time
limit 10


SELECT 
--*
DISTINCT user_id,
count(*) as cnt
FROM spc_raw_adjust.events_dafiti ed
WHERE
	partition_field BETWEEN '20200901' AND '20200901'
	AND country = 'br'
	AND user_id IS NOT NULL
group by user_id
ORDER BY cnt DESC 
	--limit 10

