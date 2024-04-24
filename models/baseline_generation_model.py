import json
import random

import torch
from loguru import logger
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, \
    BitsAndBytesConfig

from models.baseline_model import BaselineModel


class GenerationModel(BaselineModel):
    def __init__(self, config):
        super().__init__()

        # Getting parameters from the configuration file
        llm_path = config["llm_path"]
        prompt_templates_file = config["prompt_templates_file"]
        train_data_file = config["train_data_file"]
        use_quantization = config.get("use_quantization", True)

        # Generation parameters
        self.few_shot = config.get("few_shot", 5)
        self.batch_size = config.get("batch_size", 4)
        self.max_new_tokens = config.get("max_new_tokens", 64)

        # Initialize the model and tokenizer
        logger.info(f"Loading the tokenizer `{llm_path}`...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Loading the model `{llm_path}`...")
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="auto"
            )
        self.pipe = pipeline(
            task="text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
        )

        # Prompt templates
        self.prompt_templates = self.read_prompt_templates_from_csv(
            prompt_templates_file)

        # Instantiate templates with train data
        self.in_context_examples = self.instantiate_in_context_examples(
            train_data_file)

    def instantiate_in_context_examples(self, train_data_file):
        logger.info(f"Reading train data from `{train_data_file}`...")
        with open(train_data_file) as f:
            train_data = [json.loads(line) for line in f]

        # Instantiate templates with train data
        in_context_examples = []

        logger.info("Instantiating in-context examples with train data...")
        for row in train_data:
            template = self.prompt_templates[row["Relation"]]
            example = (
                f'{template.format(subject_entity=row["SubjectEntity"])} '
                f'{", ".join(row["ObjectEntities"]) if row["ObjectEntities"] else "None"}'
            )
            in_context_examples.append(example)

        return in_context_examples

    def create_prompt(self, subject_entity: str, relation: str) -> str:
        template = self.prompt_templates[relation]
        if self.few_shot > 0:
            random_examples = random.sample(
                self.in_context_examples,
                min(self.few_shot, len(self.in_context_examples))
            )
        else:
            random_examples = []
        few_shot_examples = "\n".join(random_examples)
        prompt = (f"{few_shot_examples}"
                  f"\n{template.format(subject_entity=subject_entity)}")
        return prompt

    def generate_predictions(self, inputs):
        logger.info("Generating predictions...")
        prompts = [
            self.create_prompt(
                subject_entity=inp["SubjectEntity"],
                relation=inp["Relation"]
            ) for inp in inputs
        ]

        outputs = []
        for i in tqdm(range(0, len(prompts), self.batch_size),
                      total=(len(prompts) // self.batch_size + 1),
                      desc="Generating predictions"):
            prompt_batch = prompts[i:i + self.batch_size]
            output = self.pipe(
                prompt_batch,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            outputs.extend(output)

        logger.info("Disambiguating entities...")
        results = []
        for inp, output, prompt in tqdm(zip(inputs, outputs, prompts),
                                        total=len(inputs),
                                        desc="Disambiguating entities"):
            # Remove the original prompt from the generated text
            qa_answer = output[0]["generated_text"].split(prompt)[
                -1].split("\n")[0].strip()
            wikidata_ids = self.disambiguate_entities(qa_answer)
            results.append({
                "SubjectEntityID": inp["SubjectEntityID"],
                "SubjectEntity": inp["SubjectEntity"],
                "Relation": inp["Relation"],
                "ObjectEntitiesID": wikidata_ids,
            })

        return results

    def disambiguate_entities(self, qa_answer: str):
        wikidata_ids = []
        qa_entities = qa_answer.split(", ")
        for entity in qa_entities:
            entity = entity.strip()
            if entity.startswith("and "):
                entity = entity[4:].strip()
            wikidata_id = self.disambiguation_baseline(entity)
            if wikidata_id:
                wikidata_ids.append(wikidata_id)

        return wikidata_ids
