# LM-KBC: Knowledge Base Construction from Pre-trained Language Models (3rd Edition)

This repository hosts data for the LM-KBC challenge at ISWC 2024 (https://lm-kbc.github.io/challenge2024/).

This repository contains:

- The dataset for the challenge
- Evaluation script
- Baselines
- Instructions for submitting your predictions

## Table of contents

1. [News](#news)
2. [Challenge overview](#challenge-overview)
3. [Dataset](#dataset)
4. [Evaluation metrics](#evaluation-metrics)
5. [Getting started](#getting-started)
    - [Setup](#setup)
    - [Baselines](#baselines)
        - [Baseline performance](#baseline-performance)
    - [How to structure your prediction file](#how-to-structure-your-prediction-file)

## News

- 22.4.2024: Release of dataset v1.0
- 25.3.2024: Release of preliminary evaluation script and GPT-baseline

## Challenge overview

Pretrained language models (LMs) like ChatGPT have advanced a range of semantic tasks and have also shown promise for
knowledge extraction from the models itself. Although several works have explored this ability in a setting called
probing or prompting, the viability of knowledge base construction from LMs remains under-explored. In the 3rd edition
of this challenge, we invite participants to build actual disambiguated knowledge bases from LMs, for given subjects and
relations. In crucial difference to existing probing benchmarks like
LAMA ([Petroni et al., 2019](https://arxiv.org/pdf/1909.01066.pdf)), we make no simplifying assumptions on relation
cardinalities, i.e., a subject-entity can stand in relation with zero, one, or many object-entities. Furthermore,
submissions need to go beyond just ranking predicted surface strings and materialize disambiguated entities in the
output, which will be evaluated using established KB metrics of precision and recall.

> Formally, given the input subject-entity (s) and relation (r), the task is to predict all the correct
> object-entities ({o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>k</sub>}) using LM probing.

## Dataset

Number of unique subject-entities in the data splits.

<table>
<thead>
    <tr>
        <th>Relation</th>
        <th>Train</th>
        <th>Val</th>
        <th>Test</th>
        <th>Special features</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>countryLandBordersCountry</td>
        <td>63</td>
        <td>63</td>
        <td>63</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>personHasCityOfDeath</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>seriesHasNumberOfEpisodes</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric</td>
    </tr>
    <tr>
        <td>awardWonBy</td>
        <td>10</td>
        <td>10</td>
        <td>10</td>
        <td>Many objects per subject</td>
    </tr>
    <tr>
        <td>companyTradesAtStockExchange</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
</tbody>
</table>

## Evaluation metrics

We evaluate the predictions using macro precision, recall, and F1-score.
See the evaluation script (`evaluate.py`) for more details.

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p data/testrun-XYZ.jsonl
```

Parameters: ``-g`` (the ground truth file), ``-p`` (the prediction file).

## Getting started

### Setup

1. Clone this repository:

    ```bash
    mkdir lm-kbc-2024
    cd lm-kbc-2024
    git clone https://github.com/lm-kbc/dataset2024.git
    cd dataset2024
    ```

2. Create a virtual environment and install the requirements:

    ```bash
    conda create -n lm-kbc-2024 python=3.12
    ```

    ```bash
    conda activate lm-kbc-2024
    pip install -r requirements.txt
    ```

3. Write your own solution and generate predictions (format described
   in [How to structure your prediction file](#how-to-structure-your-prediction-file)).
4. Evaluate your predictions using the evaluation script (see [Evaluation metrics](#evaluation-metrics)).
5. Submit your solutions to the organizers (see [Call for Participants](https://lm-kbc.github.io/challenge2024/#call-for-participants)).

### Baselines

As baseline, we provide a script that can run masked LMs and causal LMs from HuggingFace in the `baseline.py` script, use these
to generate entity surface forms, and use a Wikidata API for entity disambiguation.

Running instructions for the HuggingFace baselines:

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

- For OPT-1.3B

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

#### Baseline performance

TBD

### How to structure your prediction file

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntitiesID``: the predicted object entities ID, which should be a list of Wikidata IDs (strings).

You can take a look at the [example prediction file](data/dev.pred.jsonl) to
see how a valid prediction file should look like.

This is an example of how to write a prediction file:

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
