# step7_automated_image_collection.py - Comprehensive Image Collection

import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import urllib.request
from PIL import Image
import hashlib
from tqdm import tqdm
import random

class FashionImageCollector:
    def __init__(self, dataset_path="real_fashion_dataset"):
        self.dataset_path = dataset_path
        self.categories = {
            'shirts_blouses': ['white shirt', 'blue shirt', 'dress shirt', 'blouse', 'button up shirt'],
            'tshirts_tops': ['t-shirt', 'tee shirt', 'casual top', 'polo shirt', 'tank top'],
            'dresses': ['dress', 'summer dress', 'cocktail dress', 'casual dress', 'maxi dress'],
            'pants_jeans': ['jeans', 'trousers', 'pants', 'denim', 'chinos'],
            'shorts': ['shorts', 'denim shorts', 'casual shorts', 'bermuda shorts'],
            'skirts': ['skirt', 'mini skirt', 'midi skirt', 'maxi skirt', 'pencil skirt'],
            'jackets_coats': ['jacket', 'coat', 'blazer', 'windbreaker', 'puffer jacket'],
            'sweaters': ['sweater', 'hoodie', 'cardigan', 'pullover', 'sweatshirt'],
            'shoes_sneakers': ['sneakers', 'running shoes', 'casual shoes', 'athletic shoes'],
            'shoes_formal': ['dress shoes', 'formal shoes', 'heels', 'oxford shoes', 'loafers'],
            'bags_accessories': ['handbag', 'purse', 'backpack', 'tote bag', 'wallet']
        }
        
    def setup_webdriver(self):
        """Setup Chrome webdriver for image collection"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except:
            print("❌ ChromeDriver not found. Install with: pip install chromedriver-autoinstaller")
            return None
    
    def collect_from_unsplash(self, category, search_terms, target_count=80):
        """Collect high-quality images from Unsplash"""
        print(f"📸 Collecting from Unsplash: {category}")
        
        collected = 0
        for term in search_terms:
            if collected >= target_count:
                break
                
            # Unsplash API (free tier: 50 requests/hour)
            url = f"https://api.unsplash.com/search/photos"
            params = {
                'query': term,
                'per_page': 20,
                'client_id': 'Qdy3KUxyF_F5FJY3fzeXG0oqy7ixIoEoM0VC9L-WyFw'
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for i, photo in enumerate(data['results']):
                        if collected >= target_count:
                            break
                            
                        img_url = photo['urls']['regular']
                        filename = f"unsplash_{term}_{i}_{collected}.jpg"
                        
                        if self.download_image(img_url, category, filename):
                            collected += 1
                            
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Unsplash error for {term}: {e}")
                
        return collected
    
    def collect_from_pixabay(self, category, search_terms, target_count=80):
        """Collect images from Pixabay"""
        print(f"📸 Collecting from Pixabay: {category}")
        
        collected = 0
        for term in search_terms:
            if collected >= target_count:
                break
                
            # Pixabay API (free: 5000 requests/hour)
            url = "https://pixabay.com/api/"
            params = {
                'key': '50561357-5b0145b9aa7f635751989bd79',
                'q': term,
                'image_type': 'photo',
                'category': 'fashion',
                'per_page': 20
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for i, image in enumerate(data['hits']):
                        if collected >= target_count:
                            break
                            
                        img_url = image['webformatURL']
                        filename = f"pixabay_{term}_{i}_{collected}.jpg"
                        
                        if self.download_image(img_url, category, filename):
                            collected += 1
                            
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Pixabay error for {term}: {e}")
                
        return collected
    
    def collect_with_selenium(self, category, search_terms, target_count=40):
        """Collect images using Selenium (Google Images)"""
        print(f"🔍 Collecting with Selenium: {category}")
        
        driver = self.setup_webdriver()
        if not driver:
            return 0
            
        collected = 0
        
        for term in search_terms:
            if collected >= target_count:
                break
                
            try:
                # Google Images search
                search_url = f"https://www.google.com/search?q={term}&tbm=isch"
                driver.get(search_url)
                time.sleep(2)
                
                # Scroll to load more images
                for _ in range(3):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                
                # Find image elements
                images = driver.find_elements(By.CSS_SELECTOR, "img[data-src]")
                
                for i, img in enumerate(images[:15]):  # Limit per search term
                    if collected >= target_count:
                        break
                        
                    try:
                        img_url = img.get_attribute("data-src") or img.get_attribute("src")
                        if img_url and img_url.startswith('http'):
                            filename = f"google_{term}_{i}_{collected}.jpg"
                            
                            if self.download_image(img_url, category, filename):
                                collected += 1
                                
                    except Exception as e:
                        continue
                        
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                print(f"Selenium error for {term}: {e}")
        
        driver.quit()
        return collected
    
    def download_image(self, url, category, filename):
        """Download and validate image"""
        try:
            # Create paths for train/validation/test splits
            splits = ['train', 'validation', 'test']
            split_probabilities = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test
            chosen_split = random.choices(splits, weights=split_probabilities)[0]
            
            file_path = os.path.join(self.dataset_path, chosen_split, category, filename)
            
            # Download
            urllib.request.urlretrieve(url, file_path)
            
            # Validate image
            try:
                with Image.open(file_path) as img:
                    # Check minimum size
                    if img.size[0] < 100 or img.size[1] < 100:
                        os.remove(file_path)
                        return False
                    
                    # Convert to RGB and resize
                    img = img.convert('RGB')
                    img = img.resize((224, 224), Image.LANCZOS)
                    img.save(file_path, 'JPEG', quality=90)
                    
                return True
                
            except Exception:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return False
                
        except Exception as e:
            return False
    
    def collect_all_categories(self):
        """Collect images for all categories"""
        print("🚀 Starting comprehensive image collection...")
        
        total_collected = 0
        collection_report = {}
        
        for category, search_terms in self.categories.items():
            print(f"\n📂 Collecting for category: {category}")
            
            category_total = 0
            
            # Method 1: Unsplash (high quality)
            count1 = self.collect_from_unsplash(category, search_terms, 80)
            category_total += count1
            
            # Method 2: Pixabay
            count2 = self.collect_from_pixabay(category, search_terms, 80)
            category_total += count2
            
            # Method 3: Selenium (if still need more)
            if category_total < 100:
                count3 = self.collect_with_selenium(category, search_terms, 100 - category_total)
                category_total += count3
            
            collection_report[category] = category_total
            total_collected += category_total
            
            print(f"✅ {category}: {category_total} images collected")
        
        print(f"\n📊 COLLECTION COMPLETE!")
        print(f"Total images collected: {total_collected}")
        
        for category, count in collection_report.items():
            print(f"  {category}: {count}")
            
        return collection_report

# Run the collection
def run_image_collection():
    """Execute the image collection process"""
    
    print("🎯 AUTOMATED FASHION IMAGE COLLECTION")
    print("=" * 50)
    
    collector = FashionImageCollector()
    
    # Collect images
    report = collector.collect_all_categories()
    
    # Verify collection
    total_images = sum(report.values())
    
    if total_images >= 550:  # Minimum requirement
        print(f"\n✅ SUCCESS: {total_images} images collected!")
        print("🚀 Ready for model training!")
        return True
    else:
        print(f"\n⚠️ Only {total_images} images collected.")
        print("Need at least 550 for training.")
        return False

if __name__ == "__main__":
    run_image_collection()
