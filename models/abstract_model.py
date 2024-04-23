from typing import Dict, List


class AbstractModel:
    def __init__(self):
        pass

    def generate_predictions(self, inputs: List[Dict[str, str]]) -> List[
        List[str]]:
        """
        Generate predictions for the given subject entity and relation
        Args:
            inputs: A list of dictionaries containing the subject entity ("SubjectEntity"), its Wikidata ID ("SubjectEntityID") and relation ("Relation")

        Returns:
            A list of predictions (Wikidata IDs) along with their inputs (subject entity and relation)
        """
        raise NotImplementedError
