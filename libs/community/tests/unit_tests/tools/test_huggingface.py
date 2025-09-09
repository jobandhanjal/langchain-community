from unittest.mock import MagicMock
import pytest
from langchain_community.tools.huggingface import (
    HuggingFaceModelSearchTool,
    HuggingFaceDatasetSearchTool,
)

@pytest.fixture
def mock_hf_api(mocker):
    """Fixture to mock the HfApi client."""
    mock_api = MagicMock()
    # Mock the list_models and list_datasets methods
    mock_api.list_models.return_value = [
        MagicMock(modelId="gpt2", author="openai", tags=["text-generation"]),
        MagicMock(modelId="distilbert-base-uncased", author="distilbert", tags=["fill-mask"]),
    ]
    mock_api.list_datasets.return_value = [
        MagicMock(id="squad", author="stanford", tags=["question-answering"]),
        MagicMock(id="imdb", author="stanford", tags=["text-classification"]),
    ]
    # Patch the HfApi constructor to return our mock object
    mocker.patch(
        "langchain_community.tools.huggingface.HfApi", return_value=mock_api
    )
    return mock_api


def test_huggingface_model_search_tool(mock_hf_api):
    """Test the model search tool with mocked API."""
    tool = HuggingFaceModelSearchTool()
    result = tool.run("test query")

    # Assert that the list_models method was called
    mock_hf_api.list_models.assert_called_once_with(search="test query", top_k=3)

    # Assert that the output contains the mocked data
    assert "ID: gpt2" in result
    assert "Author: openai" in result
    assert "Tags: text-generation" in result
    assert "ID: distilbert-base-uncased" in result
    assert "---" in result


def test_huggingface_dataset_search_tool(mock_hf_api):
    """Test the dataset search tool with mocked API."""
    tool = HuggingFaceDatasetSearchTool()
    result = tool.run("another query")

    # Assert that the list_datasets method was called
    mock_hf_api.list_datasets.assert_called_once_with(search="another query", top_k=3)

    # Assert that the output contains the mocked data
    assert "ID: squad" in result
    assert "Author: stanford" in result
    assert "Tags: question-answering" in result
    assert "ID: imdb" in result
    assert "---" in result


def test_huggingface_model_search_no_results(mock_hf_api):
    """Test the model search tool when no results are found."""
    # Configure the mock to return an empty list for this test
    mock_hf_api.list_models.return_value = []
    
    tool = HuggingFaceModelSearchTool()
    result = tool.run("empty query")

    assert result == "No models found on the Hugging Face Hub for 'empty query'."