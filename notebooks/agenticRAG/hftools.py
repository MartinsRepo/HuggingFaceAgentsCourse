#from smolagents import Tool
#import random
#from smolagents import DuckDuckGoSearchTool
#from smolagents import Tool
#from huggingface_hub import list_models
#import duckduckgo_search

from smolagents import Tool
import random
from huggingface_hub import list_models
import datasets
from duckduckgo_search import DDGS # Import the DDGS class


class DuckDuckGoSearchTool(Tool):
    """
    Perform a DuckDuckGo web search.

    inputs:
      - query (str): the search query
      - max_results (int, optional): number of results (default: 5)
    output_type: string
    """
    name = "duckduckgo_search"
    description = (
        "Perform a DuckDuckGo web search. "
        "Pass a query string and receive the top results with titles, URLs, and snippets."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {"type": "integer", "description": "Max results to return.", "default": 5, "nullable": True}
    }
    output_type = "string"

    def forward(self, query: str, max_results: int = 5) -> str:
        results = [] # Initialize an empty list to store results
        try:
            # Use the DDGS class and its text method
            with DDGS() as ddgs:
                # The text method returns a generator, iterate over it
                for r in ddgs.text(keywords=query, max_results=max_results):
                    results.append(r)

        except Exception as e:
            return f"Error fetching DuckDuckGo results: {e}"

        if not results:
            return "No results found."

        lines = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "No title")
            url = item.get("href") or item.get("url", "") # Often 'href' for text results
            snippet = item.get("body", "") # Often 'body' for text results
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
        return "\n".join(lines)



class WeatherInfoTool(Tool):
    """
    Fetches dummy weather information for a given location.

    inputs:
      - location (str): the location name
    output_type: string
    """
    name = "weather_info"
    description = "Fetches dummy weather information for a given location."
    inputs = {"location": {"type": "string", "description": "Location name."}}
    output_type = "string"

    def forward(self, location: str) -> str:
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


class HubStatsTool(Tool):
    """
    Fetches the most downloaded model from a Hugging Face author.

    inputs:
      - author (str): Hugging Face username or organization
    output_type: string
    """
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {"author": {"type": "string", "description": "Hugging Face username or organization."}}
    output_type = "string"

    def forward(self, author: str) -> str:
        try:
            models = list_models(author=author, sort="downloads", direction=-1, limit=1)
            models = list(models)
            if not models:
                return f"No models found for author {author}."
            top_model = models[0]
            downloads = getattr(top_model, "downloads", None)
            download_info = f" with {downloads:,} downloads" if isinstance(downloads, int) else ""
            return f"Most downloaded model by {author}: {top_model.modelId}{download_info}."
        except Exception as e:
            return f"Error fetching models for {author}: {e}"

class GuestDatabaseTool(Tool):
    """
    Retrieves information about a guest.

    inputs:
      - guest_name (str): The name of the guest.
    output_type: string
    """
    name = "guest_information" # Give your tool a unique name
    description = (
        "Retrieve details and information about a specific guest "
        "based on their name."
    )
    inputs = {
        "guest_name": {"type": "string", "description": "The full name of the guest."}
    }
    output_type = "string" # Or potentially a structured type if needed

    def forward(self, guest_name: str) -> str:
        """
        Looks up and returns information for the given guest name.
        """
        # --- Replace this section with your actual logic ---
        # This is where you would implement how to find guest information.
        # For example, looking up in a database, a dictionary, etc.

        # Example placeholder logic:
        guest_data = datasets.load_dataset("agents-course/unit3-invitees", split="train")

        info = guest_data.get(guest_name)

        if info:
            return f"Information for {guest_name}: {info}"
        else:
            return f"No information found for guest: {guest_name}"
        # --- End of placeholder logic ---