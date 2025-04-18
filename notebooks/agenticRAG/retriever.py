from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
import datasets

class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."

# Assuming Tool is imported from smolagents.tools
# from smolagents.tools import Tool


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