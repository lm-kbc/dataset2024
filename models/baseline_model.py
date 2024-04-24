import csv

import requests
from loguru import logger

from models.abstract_model import AbstractModel


class BaselineModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def generate_predictions(self, inputs):
        raise NotImplementedError

    @staticmethod
    def read_prompt_templates_from_csv(file_path) -> dict:
        """Read prompt templates from a CSV file."""
        logger.info(
            f"Reading prompt templates from `{file_path}`..."
        )

        with open(file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            prompt_templates = {
                row["Relation"]: row["PromptTemplate"] for row in reader
            }
        return prompt_templates

    @staticmethod
    def disambiguation_baseline(item) -> str:
        """A simple disambiguation function that returns the Wikidata ID of an item."""
        item = str(item).strip()

        if not item or item == "None":
            return ""

        try:
            # If item can be converted to an integer, return it directly
            return str(int(item))
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
                return str(data["search"][0]["id"])
            except Exception as e:
                logger.error(f"Error getting Wikidata ID for `{item}`: {e}")
                return item
