# search_libraries.py
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
import requests
import json
from typing import List, Dict
from ollama_utils import load_api_keys

# Try to import GoogleSearch from serpapi, if it fails, set it to None
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None
    print("Warning: serpapi.GoogleSearch could not be imported. SerpAPI search will not be available.")

def duckduckgo_search(query: str, num_results: int = 5) -> List[Dict]:
    """Performs a search using DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return [{"title": result["title"], "url": result["href"]} for result in results]

def google_search(query: str, api_key: str, cse_id: str, num_results: int = 5) -> List[Dict]:
    """Performs a search using Google Custom Search API."""
    api_keys = load_api_keys()
    api_key = api_keys.get("google_api_key")
    cse_id = api_keys.get("google_cse_id")
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    return [{"title": item["title"], "url": item["link"]} for item in res.get("items", [])]

def serpapi_search(query: str, api_key: str, num_results: int = 5) -> List[Dict]:
    """Performs a search using SerpApi."""
    if GoogleSearch is None:
        print("Error: SerpAPI search is not available due to missing serpapi.GoogleSearch.")
        return []
    
    try:
        api_keys = load_api_keys()
        api_key = api_keys.get("serpapi_api_key")
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num_results
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        if not organic_results:
            print(f"SerpApi returned no results. Full response: {results}")
        return [{"title": result["title"], "url": result["link"]} for result in organic_results]
    except Exception as e:
        print(f"Error in SerpApi search: {str(e)}")
        return []

def serper_search(query: str, api_key: str, num_results: int = 5) -> List[Dict]:
    """Performs a search using Serper."""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        results = response.json()
        organic_results = results.get("organic", [])
        return [{"title": result["title"], "url": result["link"]} for result in organic_results]
    except requests.RequestException as e:
        print(f"Error in Serper search: {str(e)}")
        return []

def bing_search(query: str, api_key: str, num_results: int = 5) -> List[Dict]:
    """Performs a search using Bing Search API."""
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": num_results}
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        return [{"title": result["name"], "url": result["url"]} for result in search_results.get("webPages", {}).get("value", [])]
    except requests.RequestException as e:
        print(f"Error in Bing search: {str(e)}")
        return []