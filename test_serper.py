"""Simple script to test Serper API search and fetch full page content."""

import requests
from bs4 import BeautifulSoup
from common.shared_config import serper_api_key

# Define your query here
QUERY = "What is the capital of France?"

# Number of results
NUM_RESULTS = 3

# Step 1: Get search results from Serper
response = requests.post(
    "https://google.serper.dev/search",
    headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
    json={"q": QUERY, "num": NUM_RESULTS}
)
results = response.json()

print(f"Query: {QUERY}\n")
print("=" * 80)

# Step 2: Fetch full content from each URL
for i, item in enumerate(results.get("organic", []), 1):
    print(f"\n[{i}] {item['title']}")
    print(f"URL: {item['link']}")
    print(f"Snippet: {item.get('snippet', 'N/A')}")

    # Fetch the actual page
    try:
        page = requests.get(item['link'], timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(page.text, 'html.parser')

        # Remove script and style elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()

        # Get text content
        text = soup.get_text(separator=' ', strip=True)

        # Show first 2000 characters
        print(text)
    except Exception as e:
        print(f"Could not fetch page: {e}")

    print("\n" + "=" * 80)
