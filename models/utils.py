import csv

import requests
from loguru import logger


def read_prompt_templates_from_csv(file_path) -> dict:
    """Read prompt templates from a CSV file."""

    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {
            row["Relation"]: row["PromptTemplate"] for row in reader
        }
    return prompt_templates


def disambiguation_baseline(item):
    if not item:
        return item
    
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = (f"https://www.wikidata.org/w/api.php"
                   f"?action=wbsearchentities"
                   f"&search={item}"
                   f"&language=en"
                   f"&format=json")
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data["search"][0]["id"]
        except Exception as e:
            logger.error(f"Error getting Wikidata ID for `{item}`: {e}")
            return item
