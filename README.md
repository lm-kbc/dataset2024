# LM-KBC: Knowledge Base Construction from Pre-trained Language Models (3rd Edition)

This repository hosts data for the LM-KBC challenge at ISWC
2024 (https://lm-kbc.github.io/challenge2024/).

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
        - [Baseline 1: bert-large-cased](#baseline-1-bert-large-cased)
        - [Baseline 2: facebook/opt-1.3b](#baseline-2-facebookopt-13b)
        - [Baseline 3: meta-llama/llama-2-7b-hf](#baseline-3-meta-llamallama-2-7b-hf)
        - [Baseline 4: meta-llama/Meta-Llama-3-8B](#baseline-4-meta-llamameta-llama-3-8b)
        - [Baseline 5: meta-llama/Meta-Llama-3-8B-Instruct](#baseline-5-meta-llamameta-llama-3-8b-instruct)
    - [How to structure your prediction file](#how-to-structure-your-prediction-file)
    - [Submit your predictions to CodaLab](#submit-your-predictions-to-codalab)

## News

- 22.4.2024: Release of dataset v1.0
- 25.3.2024: Release of preliminary evaluation script and GPT-baseline

## Challenge overview

Pretrained language models (LMs) like ChatGPT have advanced a range of semantic
tasks and have also shown promise for
knowledge extraction from the models itself. Although several works have
explored this ability in a setting called
probing or prompting, the viability of knowledge base construction from LMs
remains under-explored. In the 3rd edition
of this challenge, we invite participants to build actual disambiguated
knowledge bases from LMs, for given subjects and
relations. In crucial difference to existing probing benchmarks like
LAMA ([Petroni et al., 2019](https://arxiv.org/pdf/1909.01066.pdf)), we make no
simplifying assumptions on relation
cardinalities, i.e., a subject-entity can stand in relation with zero, one, or
many object-entities. Furthermore,
submissions need to go beyond just ranking predicted surface strings and
materialize disambiguated entities in the
output, which will be evaluated using established KB metrics of precision and
recall.

> Formally, given the input subject-entity (s) and relation (r), the task is to
> predict all the correct
> object-entities ({o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>k</sub>}) using LM
> probing.

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
See the evaluation script ([evaluate.py](evaluate.py)) for more details.

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
    conda create -n lm-kbc-2024 python=3.12.1
    ```

    ```bash
    conda activate lm-kbc-2024
    pip install -r requirements.txt
    ```

3. Write your own solution and generate predictions (format described
   in [How to structure your prediction file](#how-to-structure-your-prediction-file)).
4. Evaluate your predictions using the evaluation script
   (see [Evaluation metrics](#evaluation-metrics)).
5. Submit your solutions to the organizers
   (
   see [Call for Participants](https://lm-kbc.github.io/challenge2024/#call-for-participants)),
   and/or submit your predictions to CodaLab
   (
   see [Submit your predictions to CodaLab](#submit-your-predictions-to-codalab)).

### Baselines

We provide baselines using Masked Language
Models ([models/baseline_fill_mask_model.py](models/baseline_fill_mask_model.py)),
Autoregressive Language
Models ([models/baseline_generation_model.py](models/baseline_generation_model.py)),
and Llama-3 chat models ([models/baseline_llama_3_chat_model.py](models/baseline_llama_3_chat_model.py)),

You can run these baselines via the [baseline.py](baseline.py) script and
providing it with the corresponding configuration file. We provide example
configuration files for the baselines in the [configs](configs) directory.

#### Baseline 1: bert-large-cased

Config
file: [configs/baseline-bert-large-cased.yaml](configs/baseline-bert-large-cased.yaml)

```bash
python baseline.py -c configs/baseline-bert-large-cased.yaml -i data/val.jsonl
python evaluate.py -g data/val.jsonl -p output/baseline-bert-large-cased.jsonl
```

Results:

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.300    0.000     0.000    0.000    0.000     0.000        1.000             3
companyTradesAtStockExchange    0.000    0.000     0.000    0.000    0.000     0.000        2.810             0
countryLandBordersCountry       0.632    0.452     0.487    0.628    0.464     0.534        2.132             1
personHasCityOfDeath            0.290    0.180     0.142    0.115    0.180     0.141        1.560            16
seriesHasNumberOfEpisodes       1.000    0.000     0.000    1.000    0.000     0.000        0.000           100
*** All Relations ***           0.463    0.129     0.125    0.184    0.055     0.084        1.566           120
```

#### Baseline 2: facebook/opt-1.3b

Config file: [configs/baseline-opt-1.3b.yaml](configs/baseline-opt-1.3b.yaml)

```bash
python baseline.py -c configs/baseline-opt-1.3b.yaml -i data/val.jsonl
python evaluate.py -g data/val.jsonl -p output/baseline-opt-1.3b.jsonl
```

Results:

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.740    0.000     0.001    0.286    0.001     0.003        0.700             7
companyTradesAtStockExchange    0.582    0.148     0.139    0.129    0.149     0.138        1.320            44
countryLandBordersCountry       0.528    0.282     0.225    0.322    0.245     0.278        2.191            19
personHasCityOfDeath            0.610    0.060     0.060    0.125    0.060     0.081        0.480            55
seriesHasNumberOfEpisodes       0.380    0.010     0.010    0.016    0.010     0.012        0.630            37
*** All Relations ***           0.530    0.108     0.096    0.185    0.037     0.062        1.056           162
```

#### Baseline 3: meta-llama/llama-2-7b-hf

Config
file: [configs/baseline-llama-2-7b-hf.yaml](configs/baseline-llama-2-7b-hf.yaml)

```bash
export HUGGING_FACE_HUB_TOKEN=your_token
python baseline.py -c configs/baseline-llama-2-7b-hf.yaml -i data/val.jsonl
python evaluate.py -g data/val.jsonl -p output/baseline-llama-2-7b-hf.jsonl
```

Results:

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.800    0.001     0.003    0.333    0.001     0.001        0.300             7
companyTradesAtStockExchange    0.537    0.216     0.224    0.347    0.228     0.275        0.750            29
countryLandBordersCountry       0.777    0.415     0.429    0.727    0.474     0.574        1.882            19
personHasCityOfDeath            0.500    0.170     0.163    0.250    0.170     0.202        0.680            34
seriesHasNumberOfEpisodes       0.060    0.040     0.040    0.040    0.040     0.040        0.990             2
*** All Relations ***           0.451    0.187     0.190    0.378    0.071     0.119        0.987            91
```

#### Baseline 4: meta-llama/Meta-Llama-3-8B

Config
file: [configs/baseline-llama-3-8b.yaml](configs/baseline-llama-3-8b.yaml)

```bash
export HUGGING_FACE_HUB_TOKEN=your_token
python baseline.py -c configs/baseline-llama-3-8b.yaml -i data/val.jsonl
python evaluate.py -g data/val.jsonl -p output/baseline-llama-3-8b.jsonl
```

Results:

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.900    0.000     0.000    0.000    0.000     0.000        0.100             9
companyTradesAtStockExchange    0.660    0.291     0.302    0.493    0.289     0.365        0.670            33
countryLandBordersCountry       0.777    0.531     0.529    0.771    0.617     0.686        2.309            14
personHasCityOfDeath            0.580    0.160     0.160    0.276    0.160     0.203        0.580            42
seriesHasNumberOfEpisodes       0.170    0.160     0.160    0.162    0.160     0.161        0.990             1
*** All Relations ***           0.537    0.257     0.260    0.487    0.094     0.157        1.011            99
```

#### Baseline 5: meta-llama/Meta-Llama-3-8B-Instruct

Config
file: [configs/baseline-llama-3-8b-instruct.yaml](configs/baseline-llama-3-8b-instruct.yaml)

```bash
export HUGGING_FACE_HUB_TOKEN=your_token
python baseline.py -c configs/baseline-llama-3-8b-instruct.yaml -i data/val.jsonl
python evaluate.py -g data/val.jsonl -p output/baseline-llama-3-8b-instruct.jsonl
```

Results:

```text
                              macro-p  macro-r  macro-f1  micro-p  micro-r  micro-f1  avg. #preds  #empty preds
awardWonBy                      0.360    0.039     0.057    0.414    0.024     0.046        8.700             1
companyTradesAtStockExchange    0.542    0.418     0.402    0.476    0.430     0.452        1.030            13
countryLandBordersCountry       0.960    0.627     0.650    0.949    0.755     0.841        2.294            18
personHasCityOfDeath            0.620    0.150     0.150    0.283    0.150     0.196        0.530            47
seriesHasNumberOfEpisodes       0.535    0.150     0.147    0.224    0.150     0.180        0.670            39
*** All Relations ***           0.631    0.304     0.303    0.564    0.132     0.214        1.233           118
```

### How to structure your prediction file

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntitiesID``: the predicted object entities ID, which should be a list
  of Wikidata IDs (strings).

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

fp = "./path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```

### Submit your predictions to CodaLab

TBA
