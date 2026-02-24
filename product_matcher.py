import asyncio
import logging
import urllib.parse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class ProductMatcher:
    """
    Service responsible for finding equivalent product URLs across different
    e-commerce platforms (e.g., finding the Flipkart link for an Amazon product).
    """

    async def get_flipkart_url(self, product_title: str) -> str:
        """
        Takes a product title and uses Yahoo Search
        to find the closest matching Flipkart URL.
        """
        if not product_title:
            return ""
            
        clean_title = product_title.split(',')[0].split('|')[0]
        words = clean_title.split()
        if len(words) > 8:
            clean_title = " ".join(words[:8])
            
        logger.info(f"Searching Yahoo for: {clean_title}")
        query = f"site:flipkart.com {clean_title}"
        search_url = f"https://search.yahoo.com/search?p={urllib.parse.quote_plus(query)}"
        
        try:
            def sync_search():
                headers = {
                    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
                resp = requests.get(search_url, headers=headers, timeout=10)
                resp.raise_for_status()
                return resp.text

            html = await asyncio.to_thread(sync_search)
            
            soup = BeautifulSoup(html, 'html.parser')
            for a_tag in soup.find_all('a'):
                href = a_tag.get('href', '')
                
                # Yahoo wraps external links in its own redirector
                if 'RU=' in href:
                    href = urllib.parse.unquote(href.split('RU=')[1].split('/R')[0])
                    
                if 'flipkart.com' in href and ('/p/' in href or '/product/' in href):
                    logger.info(f"Found Flipkart match: {href}")
                    # Clean the tracking parameters
                    clean_fk_url = href.split('?')[0]
                    if 'pid=' in href:
                        clean_fk_url += "?pid=" + href.split('pid=')[1][:16]
                    return clean_fk_url
                            
        except Exception as e:
            logger.error(f"Error during Yahoo Search for Flipkart URL: {str(e)}")
            print(f"CRITICAL URL MATCHER ERROR: {str(e)}")
            
        return ""
