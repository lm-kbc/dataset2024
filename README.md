# LM-KBC: Knowledge Base Construction from Pre-trained Language Models (3rd Edition)

This repository contains dataset for the 3rd LM-KBC challenge at ISWC 2024. Visit the challenge's website for more information: https://lm-kbc.github.io/challenge2024/

### Dataset v0.1

- 25.3.2024: Release of preliminary dataset v0.1, evaluation script, GPT-baseline

### Baselines

As baseline, we provide a script that can run masked LMs and causal LMs from Huggingface in the baseline.py, use these to generate entity surface forms, and use a Wikidata API for entity disambiguation.

Running instructions for the Huggingface baselines:
- For BERT

```bash
python baseline.py  \
  --input data/val.jsonl \
  --fill_mask_prompts prompts.csv \
  --question_prompts question-prompts.csv \
  --output testrun-bert.jsonl \
  --train_data data/train.jsonl \
  --model bert-large-cased \
  --batch_size 32 \
  --gpu 0
```

- For OPT-1.3b

```bash
python baseline.py \
  --input data/val.jsonl \
  --fill_mask_prompts prompts.csv \
  --question_prompts question-prompts.csv \
  --output testrun-opt.jsonl \
  --train_data data/train.jsonl \
  --model facebook/opt-1.3b \
  --batch_size 8 \
  --gpu 0
```

 
### Evaluation Script

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p data/testrun-XYZ.jsonl
```

Parameters: ``-g`` (the ground truth file), ``-p`` (the prediction file).

### Dataset Characteristics
Number of unique subject-entities in the data splits.

```text
| Relation                    |Train| |Val| |Test| Special features |
|-------------------------------------------------------------------|
| countryLandBordersCountry       63    63     63       Null values |
| personHasCityOfDeath           100   100    100       Null values |
| seriesHasNumberOfEpisodes     ...                         numeric |
| awardWonBy                    ...                 very long lists |
| companyTradesAtStockExchange  ...                     Null values |                           
```

### Baseline Performance

BERT

```text
| p   r   f1
|-----------------------------------------------------------------|
| countryLandBordersCountry     ...
| personHasCityOfDeath          ...
| seriesHasNumberOfEpisodes     ...
| awardWonBy                    ...
| companyTradesAtStockExchange  ...
| *** Average ***               ...
```

### YOUR Prediction File

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntitiesID``: the predicted object entities ID, which should be a list of Wikidata IDs (strings).

You can take a look at the [example prediction file](data/dev.pred.jsonl) to
see how a valid prediction file should look like.

This is how we write our prediction file:

```python
import json

# Dummy predictions
predictions = [
    {
        "SubjectEntity": "Dominican republic",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": ["Q790", "Q717", "Q30", "Q183"]
    },
    {
        "SubjectEntity": "Eritrea",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": ["Q115"]
    },
    {
        "SubjectEntity": "Estonia",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": []
    }

]

fp = "/path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```
