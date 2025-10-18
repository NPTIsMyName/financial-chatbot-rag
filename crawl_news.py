import requests
from bs4 import BeautifulSoup
import json
import time
import re

def get_article_links():
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MyCrawler/1.0)"
    }
    links = []
    
    # Crawl multiple pages to get around 100 articles
    for page in range(1, 6):  # Pages 1-5 should give us ~100 articles
        if page == 1:
            url = "https://vnexpress.net/kinh-doanh"
        else:
            url = f"https://vnexpress.net/kinh-doanh-p{page}"
        
        try:
            print(f"Getting links from page {page}...")
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            page_links = []
            for item in soup.select(".title-news a[href]"):
                link = item["href"]
                if link.startswith("/"):
                    link = "https://vnexpress.net" + link
                page_links.append(link)
            
            links.extend(page_links)
            print(f"Found {len(page_links)} links on page {page}")
            time.sleep(1)  # Be respectful to the server
            
        except Exception as e:
            print(f"Error getting page {page}: {e}")
            continue
    
    return list(set(links))  # loại trùng

def crawl_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MyCrawler/1.0)"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.select_one("h1.title-detail")
        date = soup.select_one("span.date")
        content = soup.select_one("article.fck_detail")

        raw_content = content.get_text(" ", strip=True) if content else None

        def extract_author_from_content(text: str):
            if not text:
                return text, None
            
            # Normalize whitespace
            normalized = re.sub(r"\s+", " ", text).strip()
            
            # Try multiple patterns to find author
            patterns = [
                # Pattern 1: Last segment after final period
                r"(.+?)\.\s*([A-Za-zÀ-ỹ][^.]{1,80})\s*$",
                # Pattern 2: Author in parentheses at the end
                r"(.+?)\s*\(([A-Za-zÀ-ỹ][^)]{1,80})\)\s*$",
                # Pattern 3: Author after "theo" or "nguồn"
                r"(.+?)\s*(?:theo|nguồn|nguồn tin)\s+([A-Za-zÀ-ỹ][^.]{1,80})\s*$",
                # Pattern 4: Author after dash or pipe
                r"(.+?)\s*[-|]\s*([A-Za-zÀ-ỹ][^.]{1,80})\s*$"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, normalized, re.IGNORECASE)
                if match:
                    content_part = match.group(1).strip()
                    author_candidate = match.group(2).strip()
                    
                    # Validate author candidate
                    if (2 <= len(author_candidate) <= 80 and 
                        re.search(r"[A-Za-zÀ-ỹ]", author_candidate) and
                        not re.search(r"\d{4}", author_candidate) and  # Not a year
                        not re.search(r"(GMT|UTC|AM|PM)", author_candidate, re.IGNORECASE)):  # Not time
                        
                        # Clean up content - remove trailing author patterns
                        content_clean = content_part
                        if content_clean and not content_clean.endswith('.'):
                            content_clean += '.'
                        return content_clean, author_candidate
            
            # Fallback: try simple last period split
            head, sep, tail = normalized.rpartition('.')
            if sep:
                author_candidate = tail.strip()
                if (2 <= len(author_candidate) <= 80 and 
                    re.search(r"[A-Za-zÀ-ỹ]", author_candidate) and
                    not re.search(r"\d{4}", author_candidate)):
                    content_only = head.strip()
                    if content_only:
                        content_only += '.'
                    return content_only, author_candidate
            
            return normalized, None

        content_text, author = extract_author_from_content(raw_content or "")

        return {
            "title": title.get_text(strip=True) if title else None,
            "date": date.get_text(strip=True) if date else None,
            "content": content_text if raw_content is not None else None,
            "author": author,
            "url": url
        }
    except Exception as e:
        print(f"Error when crawling {url}: {e}")
        return None

def main():
    print("Getting posts...")
    links = get_article_links()
    print(f"Find {len(links)} posts.")

    articles = []
    author_count = 0
    for i, link in enumerate(links, 1):
        print(f"[{i}/{len(links)}] Crawling: {link}")
        data = crawl_article(link)
        if data:
            articles.append(data)
            if data.get("author"):
                author_count += 1
                print(f"  ✓ Author found: {data['author']}")
            else:
                print(f"  ✗ No author extracted")
        time.sleep(1)  # tránh bị chặn

    with open("vnexpress_kinhdoanh.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(articles)} articles in vnexpress_kinhdoanh.json")
    print(f"Author extraction success: {author_count}/{len(articles)} ({author_count/len(articles)*100:.1f}%)")

if __name__ == "__main__":
    main()
