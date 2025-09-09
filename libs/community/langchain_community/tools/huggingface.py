from __future__ import annotations
from langchain.tools import BaseTool

class HuggingFaceHubTool(BaseTool):
    """Base tool for interacting with the Hugging Face Hub."""

    api_client: "HfApi"
    task: str = ""
    top_k_results: int = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            from huggingface_hub import HfApi
            self.api_client = HfApi()
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is not installed. Please install it with `pip install huggingface-hub`"
            ) from e

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            if self.task == "models":
                results_list = self.api_client.list_models(
                    search=query, top_k=self.top_k_results
                )
            elif self.task == "datasets":
                results_list = self.api_client.list_datasets(
                    search=query, top_k=self.top_k_results
                )
            else:
                return "Invalid task specified for the Hugging Face Hub tool."

            if not results_list:
                return f"No {self.task} found on the Hugging Face Hub for '{query}'."

            formatted_results = []
            for result in results_list:
                header = f"ID: {getattr(result, 'modelId', getattr(result, 'id', 'N/A'))}"
                author = f"Author: {getattr(result, 'author', 'N/A')}"
                tags = f"Tags: {', '.join(getattr(result, 'tags', []))}"
                formatted_results.append(f"{header}\n{author}\n{tags}")

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            return f"An error occurred: {e}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(
            None, self._run, query
        )


class HuggingFaceModelSearchTool(HuggingFaceHubTool):
    """Tool that searches for models on the Hugging Face Hub."""

    name: str = "hugging_face_model_search"
    description: str = (
        "Use this tool to search for models on the Hugging Face Hub. "
        "The input should be a search query string. "
        "The output will be a formatted string with the top results, "
        "including their ID, author, and tags."
    )
    task: str = "models"


class HuggingFaceDatasetSearchTool(HuggingFaceHubTool):
    """Tool that searches for datasets on the Hugging Face Hub."""

    name: str = "hugging_face_dataset_search"
    description: str = (
        "Use this tool to search for datasets on the Hugging Face Hub. "
        "The input should be a search query string. "
        "The output will be a formatted string with the top results, "
        "including their ID, author, and tags."
    )
    task: str = "datasets"