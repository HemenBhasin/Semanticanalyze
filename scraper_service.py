import asyncio
from playwright.async_api import async_playwright
import logging
import urllib
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import List, Dict

# Set up logging
logger = logging.getLogger(__name__)

class ScraperService:
    def __init__(self):
        self.amazon_selectors = {
            'review_block': 'div[data-hook="review"]',
            'title': 'a[data-hook="review-title"] span',
            'body': 'span[data-hook="review-body"] span',
            'rating': 'i[data-hook="review-star-rating"] span',
            'date': 'span[data-hook="review-date"]',
            'next_page': 'ul.a-pagination li.a-last a'
        }
        
    async def extract_amazon_url_from_link(self, url: str) -> str:
        """
        Takes a raw Amazon product link and converts it to the "See All Reviews" URL
        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')
            
            asin = None
            if 'dp' in path_parts:
                idx = path_parts.index('dp')
                if len(path_parts) > idx + 1:
                    asin = path_parts[idx + 1]
            elif 'gp' in path_parts and 'product' in path_parts:
                 idx = path_parts.index('product')
                 if len(path_parts) > idx + 1:
                    asin = path_parts[idx + 1]
            else:
                for part in path_parts:
                    if len(part) == 10 and part.isalnum():
                        asin = part
                        break
                        
            if not asin:
                raise ValueError("Could not extract ASIN from URL")
                
            base_idx = parsed.netloc or "www.amazon.in" 
            return f"https://{base_idx}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
        except Exception as e:
            logger.error(f"Error parsing Amazon URL: {str(e)}")
            return url
            
    async def scrape_amazon_reviews(self, url: str) -> dict:
        """
        Scrapes Amazon reviews headlessly using Playwright.
        Includes optimization to block images/CSS and diverse fetching (Recent/Critical).
        Returns a dict containing 'reviews' and 'product_title'.
        """
        reviews_data = []
        product_title = ""
        
        from bs4 import BeautifulSoup
        import urllib.parse
        from playwright.async_api import async_playwright
        
        try:
             parsed = urllib.parse.urlparse(url)
             base_product_url = f"https://{parsed.netloc}{parsed.path}"
             
             # Extract ASIN to construct direct review pages
             path = parsed.path
             asin = None
             if '/dp/' in path:
                 asin = path.split('/dp/')[1].split('/')[0]
             elif '/product/' in path:
                 asin = path.split('/product/')[1].split('/')[0]
                 
             if not asin:
                 return {'reviews': reviews_data, 'product_title': product_title}
                 
             # We want to fetch Top and Recent reviews to naturally approximate the true distribution
             target_urls = [
                 # Main product page (least likely to be blocked)
                 base_product_url,
                 # Top Reviews
                 f"https://{parsed.netloc}/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_reviews&pageNumber=1",
                 # Most Recent Reviews
                 f"https://{parsed.netloc}/product-reviews/{asin}/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=1"
             ]
        except:
             return {'reviews': reviews_data, 'product_title': product_title}
             
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox', 
                    '--disable-setuid-sandbox', 
                    '--disable-blink-features=AutomationControlled',
                    '--window-size=1920,1080',
                    '--blink-settings=imagesEnabled=false' # Hard block images
                ]
            )
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080},
                locale='en-IN'
            )
            
            # Block unnecessary resources to speed up page load globally
            async def intercept_route(route):
                if route.request.resource_type in ["image", "media", "stylesheet", "font"]:
                    await route.abort()
                else:
                    await route.continue_()
            
            async def fetch_page(url):
                page_reviews = []
                page_title = ""
                page = await context.new_page()
                await page.route("**/*", intercept_route)
                
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """)
                
                try:
                    print(f"Navigating to diverse Amazon review page: {url}")
                    # Use domcontentloaded for faster apparent load, timeout at 45s
                    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    
                    # Grab title early!
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    try:
                        # Primary: #productTitle (main page), Secondary: product-link (reviews page)
                        title_elem = soup.select_one('#productTitle, a[data-hook="product-link"], .product-title-word-break')
                        if title_elem:
                            candidate = title_elem.text.strip()
                            if len(candidate) > 5 and candidate.lower() not in ['electronics', 'computers', 'smartphones', 'clothing']:
                                page_title = candidate
                                
                        if not page_title:
                            full_page_title = soup.title.text if soup.title else ""
                            if full_page_title and "Amazon" in full_page_title:
                                cleaned = full_page_title.split(':')[-1].replace('Customer reviews', '').strip()
                                if len(cleaned) > 5 and "Amazon" not in cleaned and cleaned.lower() not in ['electronics', 'computers', 'smartphones']:
                                    page_title = cleaned
                    except:
                        pass
                    
                    # Scroll to load reviews and bypass lazy loading. 
                    # Use smaller timeouts as we rely on query_selector wait states.
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.75)")
                    await page.wait_for_timeout(500)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(500)
                    
                    # Check if Amazon blocked the page
                    try:
                        await page.wait_for_selector('div[data-hook="review"], .review', timeout=5000)
                    except Exception:
                        # It might just be the main product page fallback
                        try:
                            await page.wait_for_selector('#customerReviews', timeout=3000)
                        except:
                            print(f"Could not find reviews on page: {url}")
                            return page_reviews, page_title

                    # Click "Read more" buttons to expand long reviews
                    expand_selectors = [
                        'a[data-hook="cr-translate-these-reviews-link"]',
                        'a[data-action="cr-truncate-review-body"]',
                        'a[data-hook="review-body-read-more"]',
                        '.a-expander-prompt',
                        'a[class*="a-expander-header"]'
                    ]
                    expand_buttons = await page.query_selector_all(', '.join(expand_selectors))
                    
                    any_clicked = False
                    for btn in expand_buttons:
                        try:
                            if await btn.is_visible():
                                await btn.click(timeout=1000)
                                any_clicked = True
                        except:
                            pass
                            
                    if any_clicked:
                        await page.wait_for_timeout(300)
                            
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    review_elements = soup.select('div[data-hook="review"]')
                    if not review_elements:
                         review_elements = soup.select('.review') 
                    if not review_elements:
                         review_elements = soup.select('#cm-cr-dp-review-list div.a-section.review')

                    print(f"Found {len(review_elements)} reviews on this page.")
                    
                    for raw_review in review_elements:
                        try:
                            title_elem = raw_review.select_one('a[data-hook="review-title"] span')
                            if not title_elem:
                                 title_elem = raw_review.select_one('span[data-hook="review-title"]')
                            if not title_elem:
                                 title_elem = raw_review.select_one('.review-title-content')
                            
                            title = title_elem.text.strip() if title_elem else "No Title"
                            if "out of 5 stars" in title:
                                    parts = title.split("stars", 1)
                                    if len(parts) > 1:
                                        title = parts[1].strip()
                            
                            body_elem = raw_review.select_one('span[data-hook="review-body"]')
                            if not body_elem:
                                 body_elem = raw_review.select_one('.review-text-content')
                                 
                            if body_elem:
                                 for extraneous in body_elem.select('.video-block, .a-expander-prompt, a[data-hook="review-body-read-more"]'):
                                     extraneous.decompose()
                                 body = body_elem.get_text(separator=' ', strip=True)
                            else:
                                 body = ""
                            
                            rating_elem = raw_review.select_one('i[data-hook="review-star-rating"] span')
                            if not rating_elem:
                                 rating_elem = raw_review.select_one('.review-rating span')
                            rating_text = rating_elem.text.strip() if rating_elem else ""
                            numeric_rating = float(rating_text.split(' ')[0]) if rating_text else 0.0
                            
                            date_elem = raw_review.select_one('span[data-hook="review-date"]')
                            if not date_elem:
                                 date_elem = raw_review.select_one('.review-date')
                            date_text = date_elem.text.strip() if date_elem else ""
                            
                            full_text = f"{title}. {body}" if title != "No Title" and title not in body else body
                            
                            if len(full_text) > 5:
                                page_reviews.append({
                                    'source': 'Amazon',
                                    'rating': numeric_rating,
                                    'text': full_text,
                                    'date': date_text
                                })
                        except Exception as e:
                            logger.warning(f"Failed to parse review block: {str(e)}")
                            
                except Exception as e:
                    logger.error(f"Error scraping variant page: {str(e)}")
                finally:
                    await page.close()
                    
                return page_reviews, page_title

            try:
                import asyncio
                # Execute all target URLs horizontally in parallel tabs!
                results = await asyncio.gather(*(fetch_page(url) for url in target_urls))
                
                seen_texts = set()
                # Aggregate results synchronously
                for page_reviews, page_title in results:
                    if not product_title and page_title:
                        product_title = page_title
                        
                    for rev in page_reviews:
                        if rev['text'] not in seen_texts:
                            seen_texts.add(rev['text'])
                            reviews_data.append(rev)
                            
            except Exception as e:
                logger.error(f"Error running Amazon scraper: {str(e)}")
            finally:
                await browser.close()
                
        return {'reviews': reviews_data, 'product_title': product_title}

    async def scrape_flipkart_reviews(self, url: str) -> List[Dict]:
        """
        Scrapes Flipkart reviews from the dedicated reviews page using Playwright,
        fetching diverse review types (Recent, Most Helpful, Critical).
        """
        reviews_data = []
        
        # Extract product slug and pid to construct the reviews page URL
        # Flipkart product URL format: flipkart.com/product-name/p/itm...
        # Reviews URL format: flipkart.com/product-name/product-reviews/itm...
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            
            # Find the /p/ segment and replace with /product-reviews/
            product_slug = path_parts[0] if path_parts else ''
            product_id = ''
            for i, part in enumerate(path_parts):
                if part == 'p' and i + 1 < len(path_parts):
                    product_id = path_parts[i + 1]
                    break
            
            if not product_id:
                print(f"Could not extract product ID from Flipkart URL: {url}")
                return reviews_data
                
            # Construct diverse review URLs with different sort orders  
            base_reviews_url = f"https://www.flipkart.com/{product_slug}/product-reviews/{product_id}"
            target_url = f"{base_reviews_url}?sortOrder=MOST_RECENT&page=1"
            
        except Exception as e:
            print(f"Error constructing Flipkart review URLs: {e}")
            target_url = url  # Fallback to original URL
        
        async with async_playwright() as p:
            iphone = p.devices['iPhone 13']
            
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox', 
                    '--disable-setuid-sandbox', 
                    '--disable-blink-features=AutomationControlled',
                    '--incognito',
                    '--disable-web-security'
                ]
            )
            
            context = await browser.new_context(
                **iphone,
                locale='en-IN',
                timezone_id='Asia/Kolkata',
                permissions=['geolocation']
            )
            
            page = await context.new_page()
            
            # Heavy stealth injection
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = { runtime: {} };
            """)
            
            seen_reviews = set()
            star_counts = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
            
            try:
                print(f"Navigating to Flipkart review page: {target_url}")
                await page.goto(target_url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(4000)
                
                # Scroll to load lazy content
                for _ in range(3):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(1500)
                
                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                print("Extracting actual star distribution...")
                # Find the distribution bars (usually list items in the rating block)
                try:
                    # Look for elements containing "★" or generic distribution indicators
                    rating_block = soup.find('div', string=lambda t: t and 'Ratings' in t and 'Reviews' in t)
                    if rating_block and rating_block.parent:
                        parent = rating_block.parent
                        counts = parent.find_all('div', string=lambda t: t and t.replace(',','').isdigit())
                        if len(counts) >= 5:
                            # Usually ordered 5, 4, 3, 2, 1
                            star_counts[5] = int(counts[0].text.replace(',',''))
                            star_counts[4] = int(counts[1].text.replace(',',''))
                            star_counts[3] = int(counts[2].text.replace(',',''))
                            star_counts[2] = int(counts[3].text.replace(',',''))
                            star_counts[1] = int(counts[4].text.replace(',',''))
                except Exception as e:
                    print(f"Could not parse star distribution, falling back to all available: {e}")
                
                total_ratings = sum(star_counts.values())
                target_ratios = {k: (v/total_ratings if total_ratings > 0 else 0) for k, v in star_counts.items()}
                
                # Fetch dynamically across a few pages to get enough raw reviews to sample from
                raw_reviews = []
                
                # Fetch BOTH Helpful and Recent to get a large pool
                urls_to_pool = [
                    f"{base_reviews_url}?sortOrder=MOST_RECENT&page=1",
                    f"{base_reviews_url}?sortOrder=MOST_HELPFUL&page=1",
                    f"{base_reviews_url}?sortOrder=MOST_RECENT&page=2"
                ]
                
                for pool_url in urls_to_pool:
                    await page.goto(pool_url, wait_until="domcontentloaded", timeout=45000)
                    await page.wait_for_timeout(3000)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)
                    
                    pool_soup = BeautifulSoup(await page.content(), 'html.parser')
                    rating_badges = pool_soup.find_all('div', string=lambda text: text and text.strip().replace('.','',1).isdigit() and 0 < float(text.strip()) <= 5.0)
                    if not rating_badges:
                        rating_badges = pool_soup.select('div._3LWZlK')
                        
                    for badge in rating_badges:
                        try:
                            numeric_rating = float(badge.text.strip())
                            if numeric_rating == 0: continue
                            
                            container = badge
                            for _ in range(6):
                                container = container.parent
                                if not container: break
                                if len(container.text.strip()) > 30: break
                                
                            if not container: continue
                            
                            all_text = container.text.strip()
                            if len(all_text) > 15 and all_text not in seen_reviews:
                                paragraphs = container.find_all('p')
                                spans = container.find_all('span')
                                text_blocks = [t.text.strip() for t in paragraphs + spans if len(t.text.strip()) > 5]
                                
                                full_text = " ".join(set(text_blocks)) if text_blocks else all_text.replace(badge.text.strip(), '', 1).strip()
                                
                                if "READ MORE" in full_text:
                                    full_text = full_text.split("READ MORE")[0].strip()
                                
                                date_text = "Unknown"
                                import re
                                date_patterns = [
                                    r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+\d{4}',
                                    r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+\d{4}',
                                    r'\d+\s+(month|day|year|week)s?\s+ago',
                                ]
                                for pattern in date_patterns:
                                    match = re.search(pattern, str(container), re.IGNORECASE)
                                    if match:
                                        date_text = match.group(0)
                                        break
                                
                                if len(full_text) > 10:
                                    raw_reviews.append({
                                        'source': 'Flipkart',
                                        'rating': numeric_rating,
                                        'text': full_text,
                                        'date': date_text
                                    })
                                    seen_reviews.add(all_text)
                        except Exception:
                            pass
                            
                # Apply Proportional Sampling if we successfully extracted the distribution
                target_total = 25 # Number of Flipkart reviews we want to show
                if total_ratings > 0 and len(raw_reviews) > 0:
                    print(f"Applying Target Ratios: {target_ratios}")
                    sampled_reviews = []
                    
                    # Group acquired reviews by star
                    grouped = {5: [], 4: [], 3: [], 2: [], 1: []}
                    for r in raw_reviews:
                        star_bucket = int(r['rating'])
                        if star_bucket in grouped:
                            grouped[star_bucket].append(r)
                            
                    for star, target_pct in target_ratios.items():
                        count_needed = int(round(target_total * target_pct))
                        # Take up to count_needed from what we scraped
                        sampled_reviews.extend(grouped[star][:count_needed])
                        
                    reviews_data = sampled_reviews
                else:
                    # Fallback if UI parsing for star counts failed
                    reviews_data = raw_reviews[:target_total]
                    
            except Exception as e:
                logger.error(f"Error scraping Flipkart review page: {str(e)}")
            finally:
                await browser.close()
                
        return reviews_data

# Quick test block
if __name__ == "__main__":
    pass
    # async def main():
    #     scraper = ScraperService()
    #     
    #     # Test Amazon
    #     sample_amazon = "https://www.amazon.in/Apple-iPhone-13-128GB-Starlight/dp/B09G9D8KRQ"
    #     print("Starting Amazon scrape...")
    #     reviews = await scraper.scrape_amazon_reviews(sample_amazon)
    #     print(f"Scraped {len(reviews)} reviews.")
    #     for r in reviews[:2]:
    #         print(f"--- {r['rating']} Stars ---")
    #         print(f"Date: {r['date']}")
    #         print(f"Text: {r['text'][:150]}...\n")
    #         
    #     # Test Flipkart
    #     # sample_flipkart = "https://www.flipkart.com/apple-iphone-12-blue-128-gb/p/itm02853ae92e90f"
    #     # print("Starting Flipkart scrape...")
    #     # fk_reviews = await scraper.scrape_flipkart_reviews(sample_flipkart)
    #     # print(f"Scraped {len(fk_reviews)} reviews.")
    #     # for r in fk_reviews[:2]:
    #     #     print(f"--- {r['rating']} Stars ---")
    #     #     print(f"Date: {r['date']}")
    #     #     print(f"Text: {r['text'][:150]}...\n")
    #         
    # asyncio.run(main())
