import requests
import pandas as pd
import logging
import os
import time

os.makedirs('Universal_Engine_Workspace/mas_logs', exist_ok=True)
logging.basicConfig(
    filename='Universal_Engine_Workspace/mas_logs/gdelt_macro.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataEngineer_GDELT')

class GDELTMacroScraper:
    def __init__(self):
        # We use the v2 API doc/doc endpoint with mode=timelinetone
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch_macro_sentiment(self, query="india", timespan="5y"):
        """
        Hits the free GDELT API to generate a daily macro sentiment_score 
        to act as an independent column next to company-specific sentiment.
        """
        logger.info(f"Initiating GDELT Macro Sentiment Scrape. Query: '{query}', Timespan: {timespan}")
        
        # GDELT has a maximum timespan of 3 years (36m) for timelines technically, 
        # but let's query the 3-year trailing window repeatedly or rely on 1y chunks if needed.
        # We will attempt a 3-year query first (36m) as 5y might get rejected by their strict API.
        
        # NOTE: GDELT limits 'timespan' for some queries. Let's do maximum safe range (3y)
        api_timespan = "36m" if timespan == "5y" else timespan
        
        params = {
            "query": query,
            "mode": "timelinetone",
            "format": "json",
            "timespan": api_timespan
        }
        
        try:
            logger.info(f"Pinging GDELT endpoint: {self.base_url}")
            response = requests.get(self.base_url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if "timeline" in data and len(data["timeline"]) > 0:
                    series = data["timeline"][0].get('data', [])
                    
                    if not series:
                        logger.warning("Empty data array returned from GDELT.")
                        return pd.DataFrame()
                        
                    df = pd.DataFrame(series)
                    
                    # Convert GDELT '20230301T000000Z' string to pd datetime
                    df['date'] = pd.to_datetime(df['date'])
                    
                    df.rename(columns={'value': 'Macro_Sentiment_Score'}, inplace=True)
                    df['date'] = df['date'].dt.tz_localize(None).dt.date
                    
                    # Log the output shape
                    logger.info(f"Successfully fetched {len(df)} macro sentiment records. Date bounds: {df['date'].min()} to {df['date'].max()}")
                    return df[['date', 'Macro_Sentiment_Score']]
                else:
                    logger.warning("JSON structure did not contain 'timeline'.")
                    return pd.DataFrame()
            else:
                logger.error(f"GDELT API error: {response.status_code} - {response.text}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error executing GDELT API request: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    scraper = GDELTMacroScraper()
    macro_df = scraper.fetch_macro_sentiment(query="india", timespan="5y") # GDELT max is roughly 3y easily
    
    if not macro_df.empty:
        output_file = 'Universal_Engine_Workspace/scripts_scraping/gdelt_macro_sentiment.csv'
        macro_df.to_csv(output_file, index=False)
        print(f"GDELT Macro Sentiment Data successfully written to {output_file}")
        print(macro_df.tail())
    else:
        print("GDELT Scraping failed or returned empty.")
