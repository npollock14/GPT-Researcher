from googlesearch import search
from utils import fetch_site_content
import json

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