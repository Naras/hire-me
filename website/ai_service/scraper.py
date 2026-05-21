import requests
from bs4 import BeautifulSoup
import re

def extract_job_description(url: str) -> str:
    """
    Fetches the content from the given URL and extracts readable text
    from the HTML page using BeautifulSoup.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script, style, header, footer, and nav elements to clean up the text
        for element in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            element.extract()
            
        # Extract text and replace multiple spaces/newlines with a single space/newline
        text = soup.get_text(separator="\n")
        
        # Clean up empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        
        return clean_text
    except Exception as e:
        print(f"Error fetching job description from {url}: {e}")
        return ""

if __name__ == "__main__":
    url = "https://sentry.io/careers/33fa0a52-0af7-4d90-8ad4-a87f47e7e5e2"
    print(extract_job_description(url)[:500])
