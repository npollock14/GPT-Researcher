from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass

@dataclass
class ModelData:
    official_name: str
    max_context: int
    pricing: dict

    # def __str__(self):
    #     return self.official_name

class ModelEnum(Enum):
    GPT4_8K = ModelData(official_name="gpt-4", max_context=8000, pricing={"input": 0.03, "output": 0.06})
    GPT4_32K = ModelData(official_name="gpt-4-32k", max_context=32000, pricing={"input": 0.06, "output": 0.12})
    GPT3_5_TURBO_4K = ModelData(official_name="gpt-3.5-turbo", max_context=4000, pricing={"input": 0.0015, "output": 0.002})
    GPT3_5_TURBO_16K = ModelData(official_name="gpt-3.5-turbo-16k", max_context=16000, pricing={"input": 0.003, "output": 0.004})

    def __str__(self):
        return self.value.official_name


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
    content: str
    cost: float = Field(..., description="Estimated cost of processing the search result")
    model: ModelEnum = Field(..., description="Model to use for processing the search result")

    def __str__(self):
        return f"Title: {self.title}\nUrl: {self.link}\nContent: {self.content}\nCost: ${self.cost}\nSummary Model: {self.model}\n\n"
    
    def to_json(self):
        return {
            "title": self.title,
            "link": self.link,
            "content": self.content,
            "cost": self.cost,
            "model": self.model.value.official_name
        }
    
if __name__ == "__main__":
    # test SearchResultsSchema
    search_result = SearchResultSchema(
        title="Test Title",
        link="https://www.google.com",
        content="Test Content",
        cost=0.1,
        model=ModelEnum.GPT4_8K
    )
    import json
    results = [search_result]
    print(json.dumps([search_result.to_json() for search_result in results], indent=4))
    # save to file
    with open('./outputs/search_results.json', 'w') as outfile:
        json.dump([search_result.to_json() for search_result in results], outfile, indent=4)