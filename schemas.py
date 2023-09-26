from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass

@dataclass
class ModelData:
    official_name: str
    max_context: int
    pricing: dict

class ModelEnum(Enum):
    GPT4_8K = ModelData(official_name="gpt-4", max_context=8000, pricing={"input": 0.03, "output": 0.06})
    GPT4_32K = ModelData(official_name="gpt-4-32k", max_context=32000, pricing={"input": 0.06, "output": 0.12})
    GPT3_5_TURBO_4K = ModelData(official_name="gpt-3.5-turbo", max_context=4000, pricing={"input": 0.0015, "output": 0.002})
    GPT3_5_TURBO_16K = ModelData(official_name="gpt-3.5-turbo-16k", max_context=16000, pricing={"input": 0.003, "output": 0.004})


class ResearchActionPlanSchema(BaseModel):
    topic_of_research: str
    purpose_of_context: str
    primary_audience: str
    type_of_paper_to_be_written: str
    paper_structure: list[str]
    research_plan: list[str]
    search_queries: list[str]

class SearchResultSchema(BaseModel):
    title: str
    link: str
    description: str
    keywords: list[str]
    summary: str
    content: str
    cost: float = Field(..., description="Estimated cost of processing the search result")

    def __str__(self):
        return f"{self.title}\n{self.link}\n{self.description}\n{self.summary}\n{self.keywords}\n${self.cost}\n\n"

    def to_prompt(self, source_number):
        return f"{source_number}:\n{self.title}\n{self.link}\n{self.description}\n{self.summary}\n{self.keywords}\n${self.cost}\n\n"

if __name__ == "__main__":
    # test SearchResultsSchema
    search_result = SearchResultSchema(
        title="Test Title",
        link="https://www.google.com",
        description="Test Description",
        keywords=["test", "keywords"],
        summary="Test Summary",
        content="Test Content",
        cost=0.1
    )
    print(search_result.to_prompt(1))

    # Example Usage:
    print(ModelEnum.GPT4_8K.value.official_name)  # Outputs: gpt-4
    print(ModelEnum.GPT4_8K.value.max_context)    # Outputs: 8000
    print(ModelEnum.GPT4_8K.value.pricing['input'])  # Outputs: 0.03
