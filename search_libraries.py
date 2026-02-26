# search_libraries.py
import requests
import json
from typing import List, Dict
from ollama_utils import load_api_keys

# Make duckduckgo_search optional to avoid dependency issues
try:
    from duckduckgo_search import DDGS
    duckduckgo_available = True
    print("CHECKPOINT: DuckDuckGo search module successfully loaded")
except ImportError as e:
    print(f"CHECKPOINT: DuckDuckGo search module not available: {str(e)}")
    duckduckgo_available = False

# Make Google Custom Search API optional
try:
    from googleapiclient.discovery import build
    google_cse_available = True
    print("CHECKPOINT: Google Custom Search API module successfully loaded")
except ImportError as e:
    print(f"CHECKPOINT: Google Custom Search API module not available: {str(e)}")
    google_cse_available = False

# Try to import GoogleSearch from serpapi, if it fails, set it to None
try:
    from serpapi import GoogleSearch
    serpapi_available = True
    print("CHECKPOINT: SerpAPI module successfully loaded")
except ImportError:
    GoogleSearch = None
    serpapi_available = False
    print("CHECKPOINT: serpapi.GoogleSearch could not be imported. SerpAPI search will not be available.")

def duckduckgo_search(query: str, num_results: int = 5) -> List[Dict]:
    """Performs a search using DuckDuckGo."""
    # Check if DuckDuckGo search is available
    if not duckduckgo_available:
        print("CHECKPOINT: DuckDuckGo search is not available. Please install the required dependency: pip install duckduckgo-search")
        return [{"title": "DuckDuckGo search is not available", 
                "url": "#", 
                "error": "Please install the required dependency: pip install duckduckgo-search"}]
    
    try:
        print(f"CHECKPOINT: Performing DuckDuckGo search for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        return [{"title": result["title"], "url": result["href"]} for result in results]
    except Exception as e:
        print(f"CHECKPOINT: Error performing DuckDuckGo search: {str(e)}")
        return [{"title": "Error performing DuckDuckGo search", 
                "url": "#", 
                "error": str(e)}]

def google_search(query: str, api_key: str = None, cse_id: str = None, num_results: int = 5) -> List[Dict]:
    """Performs a search using Google Custom Search API."""
    # Check if Google Custom Search API is available
    if not google_cse_available:
        print("CHECKPOINT: Google Custom Search API is not available. Please install the required dependency: pip install google-api-python-client")
        return [{"title": "Google Custom Search API is not available", 
                "url": "#", 
                "error": "Please install the required dependency: pip install google-api-python-client"}]
    
    try:
        print(f"CHECKPOINT: Performing Google Custom Search for: {query}")
        api_keys = load_api_keys()
        api_key = api_key or api_keys.get("google_api_key")
        cse_id = cse_id or api_keys.get("google_cse_id")
        
        if not api_key or not cse_id:
            print("CHECKPOINT: Google API key or CSE ID not found in API keys")
            return [{"title": "Google API key or CSE ID not found", 
                    "url": "#", 
                    "error": "Please set up your Google API key and CSE ID"}]
        
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return [{"title": item["title"], "url": item["link"]} for item in res.get("items", [])]
    except Exception as e:
        print(f"CHECKPOINT: Error performing Google Custom Search: {str(e)}")
        return [{"title": "Error performing Google Custom Search", 
                "url": "#", 
                "error": str(e)}]

def serpapi_search(query: str, api_key: str = None, num_results: int = 5) -> List[Dict]:
    """Performs a search using SerpApi."""
    # Check if SerpAPI is available
    if not serpapi_available:
        print("CHECKPOINT: SerpAPI search is not available. Please install the required dependency: pip install google-search-results")
        return [{"title": "SerpAPI search is not available", 
                "url": "#", 
                "error": "Please install the required dependency: pip install google-search-results"}]
    
    try:
        print(f"CHECKPOINT: Performing SerpAPI search for: {query}")
        api_keys = load_api_keys()
        api_key = api_key or api_keys.get("serpapi_api_key")
        
        if not api_key:
            print("CHECKPOINT: SerpAPI API key not found in API keys")
            return [{"title": "SerpAPI API key not found", 
                    "url": "#", 
                    "error": "Please set up your SerpAPI API key"}]
        
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
        print(f"CHECKPOINT: Error performing SerpAPI search: {str(e)}")
        return [{"title": "Error performing SerpAPI search", 
                "url": "#", 
                "error": str(e)}]

def serper_search(query: str, api_key: str = None, num_results: int = 5) -> List[Dict]:
    """Performs a search using Serper."""
    try:
        print(f"CHECKPOINT: Performing Serper search for: {query}")
        api_keys = load_api_keys()
        api_key = api_key or api_keys.get("serper_api_key")
        
        if not api_key:
            print("CHECKPOINT: Serper API key not found in API keys")
            return [{"title": "Serper API key not found", 
                    "url": "#", 
                    "error": "Please set up your Serper API key"}]
        
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": num_results
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        results = response.json()
        organic_results = results.get("organic", [])
        if not organic_results:
            print(f"CHECKPOINT: Serper returned no results for query: {query}")
        return [{"title": result["title"], "url": result["link"]} for result in organic_results]
    except requests.RequestException as e:
        print(f"CHECKPOINT: Error performing Serper search: {str(e)}")
        return [{"title": "Error performing Serper search", 
                "url": "#", 
                "error": str(e)}]

def bing_search(query: str, api_key: str = None, num_results: int = 5) -> List[Dict]:
    """Performs a search using Bing Search API."""
    try:
        print(f"CHECKPOINT: Performing Bing search for: {query}")
        api_keys = load_api_keys()
        api_key = api_key or api_keys.get("bing_api_key")
        
        if not api_key:
            print("CHECKPOINT: Bing API key not found in API keys")
            return [{"title": "Bing API key not found", 
                    "url": "#", 
                    "error": "Please set up your Bing API key"}]
        
        endpoint = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "count": num_results}
        
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        results = search_results.get("webPages", {}).get("value", [])
        
        if not results:
            print(f"CHECKPOINT: Bing returned no results for query: {query}")
            
        return [{"title": result["name"], "url": result["url"]} for result in results]
    except requests.RequestException as e:
        print(f"CHECKPOINT: Error performing Bing search: {str(e)}")
        return [{"title": "Error performing Bing search", 
                "url": "#", 
                "error": str(e)}]