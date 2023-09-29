from utils import parse_raw_summary
from schemas import SearchResultSummary, SearchResultSchema, ModelEnum

def test_parse_valid_summary():
    raw_summary = """
    Author(s): John Doe, Jane Smith
    Date: 2023-09-29
    - Detail 1
    - Detail 2
    Relevancy:
    - Section1: 5
    - Section2: 3
    """
    search_result: SearchResultSchema = SearchResultSchema(
        title="Test Title",
        link="https://www.google.com",
        content="Test Content",
        cost=0.1,
        model=ModelEnum.GPT3_5_TURBO_16K
    )
    result:SearchResultSummary = parse_raw_summary(raw_summary, search_result)
    assert result.source_material.title == "Test Title"
    assert result.details == ["Detail 1", "Detail 2"]
    assert result.relevancy == {"Section1": 5, "Section2": 3}
    assert result.error is None

def test_parse_missing_sections():
    raw_summary = """
    Title: Test Title
    Author(s): John Doe, Jane Smith
    Date: 2023-09-29
    """
    search_result: SearchResultSchema = SearchResultSchema(
        title="Test Title",
        link="https://www.google.com",
        content="Test Content",
        cost=0.1,
        model=ModelEnum.GPT3_5_TURBO_16K
    )
    result = parse_raw_summary(raw_summary, search_result)
    assert result.error == "Details section is missing or empty. Relevancy section is missing or empty."
