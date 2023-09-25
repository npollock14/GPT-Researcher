from googlesearch import search
from utils import fetch_site_content
import json
import openai  # for OpenAI API calls
from dotenv import load_dotenv
import os

# load the .env file
load_dotenv()

# set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Example of an OpenAI ChatCompletion request with stream=True
# https://platform.openai.com/docs/guides/chat

# send a ChatCompletion request to count to 100
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "user",
#             "content": "Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...",
#         }
#     ],
# )

# get the response content
# response['choices'][0]['message']['content']

def outline():
    """
    1. Prompt user for a topic to research and parse the prompt

        Example prompt:
        1. I have an upcoming business meeting with Deloitte. Create a business focused briefing for me.

        Parsed prompt:
        1. **What is the topic of research?**
        - Creating a business-focused briefing for an accountant from Deloitte who has an upcoming meeting with Taylor Morrison.

        2. **For what purpose or context is this information needed?**
        - Preparation for an upcoming business meeting with Taylor Morrison.

        3. **Who is the primary audience for the information?**
        - An accountant from Deloitte.

    2. Create a plan of action for an initial set of queries to gain background information:

    - **Research about the companies:** Gather data about both Deloitte and Taylor Morrison to understand their backgrounds, services/products they offer, and their market positioning.
    - **Recent News:** Check for any recent events or news related to the companies, which can be relevant during the meeting.
    - **Competition Analysis:** Identify and understand the major competitors for both companies.
    - **Previous Interactions:** If possible, search for any prior interactions or business deals between the two companies.
    - **Sectorial Analysis:** Understand the sectors in which the companies operate, along with current trends and forecasts.
        
    3. Create a list of initial search queries from the plan of action:
        
        - Deloitte company profile
        - Taylor Morrison company profile
        - Recent news about Deloitte
        - Recent news about Taylor Morrison
        - Major competitors of Deloitte
        - Major competitors of Taylor Morrison
        - Deloitte and Taylor Morrison past collaborations or business interactions
        - Current trends in consulting sector (for Deloitte)
        - Current trends in homebuilding or real estate sector (for Taylor Morrison)
        
    4. Search for and summarize the content learned using google search python API

        - toy with scanning the text like a human would, look at first sentence of all paragraphs and can decide what to read from there - dont want to read the whole thing
        
    5.  Assess if the data collected fulfills the initial plan of action.

        - If not, generate more search queries and repeat steps 4 and 5.
        - If yes, proceed

    6. Assess if I have enough information to complete the research paper.

        - If not, generate another plan of action that addresses the gaps in information. Loop until I have enough information or a limit is reached.
        - If yes, proceed

    7. Generate a research paper using the data collected.

        - use markdown or latex to generate a pdf file

    """

def main():
    res = list(search("Who is the ceo of Deloitte?", num_results=1, advanced=True))
    for search_result in enumerate(res):
        # pretty print the result - title, url and description
        print("Result #{}".format(search_result[0] + 1))
        print("Title: {}".format(search_result[1].title))
        print("URL: {}".format(search_result[1].url))
        print("Description: {}".format(search_result[1].description))
        print()

    # ask user what result they want to see
    resultNumber = int(input("Which result do you want to see? "))
    # fetch the content of the page
    content = fetch_site_content(res[resultNumber - 1].url)
    result = {}
    result['title'] = content.title
    result['authors'] = content.authors
    result['publish_date'] = content.publish_date
    result['text'] = content.text
    # save the content to a json file
    with open("./outputs/result.json", "w") as f:
        f.write(json.dumps(result))

if __name__ == "__main__":
    main()