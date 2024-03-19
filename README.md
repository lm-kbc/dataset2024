# Knowledge Base Construction from Pre-trained Language Models (LM-KBC) 2nd Edition

This repository contains dataset for the LM-KBC challenge at ISWC 2024.

## Dataset v0.1

 - 25.3.2024: Release of preliminary dataset v0.1, evaluation script, GPT-baseline

### Baselines

As baseline, we provide a script that can run masked LMs and causal LMs from Huggingface in the baseline.py, use these to generate entity surface forms, and use a Wikidata API for entity disambiguation.

Running instructions for the Huggingface baselines:
 - For BERT

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-bert.jsonl --train_data data/train.jsonl --model bert-large-cased --batch_size 32 --gpu 0```

 - For OPT-1.3b

```python baseline.py  --input data/val.jsonl --fill_mask_prompts prompts.csv --question_prompts question-prompts.csv  --output testrun-opt.jsonl --train_data data/train.jsonl --model facebook/opt-1.3b --batch_size 8 --gpu 0```

 
### Evaluation script

Run instructions evaluation script:
  * ```python evaluate.py -p data/val.jsonl -g data/testrun-XYZ.jsonl```

The first parameter hereby indicates the prediction file, the second the ground truth file.

----------------------------------------------------------------

## Dataset Characteristics
Number of unique subject-entities in the data splits.

```text
| Relation                    |Train| |Val| |Test| Special features |
|-------------------------------------------------------------------|
| countryLandBordersCountry       63    63     63       Null values |
| personHasCityOfDeath          100   100    100       Null values |
| seriesHasNumberOfEpisodes     ...                         numeric |
| awardWonBy                    ...                 very long lists |
| companyTradesAtStockExchange  ...                     Null values |                           
```

## Baseline performance

BERT

```text
| p   r   f1
|-----------------------------------------------------------------|
| countryLandBordersCountry         ...
| personHasCityOfDeath          ...
| seriesHasNumberOfEpisodes     ...
| awardWonBy                ...
| companyTradesAtStockExchange  ...
| *** Average ***               ...
```
----------------------------------------------------------------

### YOUR prediction file

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

# Fake predictions
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