from enum import Enum

from models.baseline_fill_mask_model import FillMaskModel
from models.baseline_generation_model import GenerationModel


class Models(Enum):
    BASELINE_FILL_MASK = "baseline_fill_mask"
    BASELINE_GENERATION = "baseline_generation"
    # Add more models here

    @staticmethod
    def get_model(model_name: str):
        model = Models(model_name)
        if model == Models.BASELINE_FILL_MASK:
            return FillMaskModel
        if model == Models.BASELINE_GENERATION:
            return GenerationModel
        else:
            raise ValueError(f"Model `{model_name}` not found.")
