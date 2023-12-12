from google_play_scraper import Sort, reviews
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import pandas as pd

def extract_values_from_url(url):
	parsed_url = urlparse(url)
	query_params = parse_qs(parsed_url.query)

	app_id = query_params.get('id', [''])[0]
	language = query_params.get('hl', [''])[0]
	country = query_params.get('gl', [''])[0]
	
	return crawling(app_id, language, country)

def crawling(app_id, language, country):
	result, _ = reviews(
			app_id, lang=language, country=country,\
			sort=Sort.MOST_RELEVANT,count=100000,filter_score_with=None
					)
	df = pd.DataFrame(result)
	df_copy = df[['at','score','content']]
	df_copy.loc[:,'at'] = pd.to_datetime(df_copy['at']).dt.strftime('%Y-%m')
	df_copy.to_csv(f'{app_id}.csv', encoding='utf-8-sig')

	return df_copy
input_url = "https://play.google.com/store/apps/details?id=com.kakao.talk&hl=kr&gl=US"
extract_values_from_url(input_url)

