# Ruler

## Prepare Environment

First, you should set up a python environment. This code base has been tested under python 3.x, and we officially support python 3.10.

```bash
conda create -n ruler python=3.10
cd Ruler # where contains 'requirements.txt'
pip install -r requirements.txt

export PYTHONPATH=xxxx/Ruler/src
cd src

# create folders and download datasets
bash ../scripts/download.sh
```

<!-- TOC -->

- [Ruler](#ruler)
  - [Prepare Environment](#prepare-environment)
  - [Target Length Generation Task](#target-length-generation-task)
    - [Select Single-Turn Dialogue Data](#select-single-turn-dialogue-data)
    - [Build Target Length Generation Dataset](#build-target-length-generation-dataset)
    - [Run on Target Length Generation Task](#run-on-target-length-generation-task)
      - [Closed-source Model](#closed-source-model)
      - [Open-source Model](#open-source-model)
    - [Calculate Scores of Target Length Generation Task](#calculate-scores-of-target-length-generation-task)
  - [Ruler](#ruler)
    - [Build Ruler Training Dataset](#build-ruler-training-dataset)
    - [Finetune the Models](#finetune-the-models)
    - [TLG Results](#tlg-results)
  - [Multi MLT Generation Experiment](#multi-mlt-generation-experiment)
    - [Build Dataset](#build-dataset)
    - [Run the Experiment](#run-the-experiment)
    - [Calculate the Metrics](#calculate-the-metrics)
  - [Self-generated MLT Experiment](#self-generated-mlt-experiment)
    - [Build Dataset](#build-dataset)
    - [Run the Experiment](#run-the-experiment)
    - [Calculate the Metrics](#calculate-the-metrics)
  - [Overall Performance](#overall-performance)
    - [Vanilla SFT](#vanilla-sft)
    - [Overall Performance Test](#overall-performance-test)

<!-- /TOC -->

## Target Length Generation Task

### Select Single-Turn Dialogue Data

We utilize [`OpenHermes-2.5`](https://huggingface.co/datasets/teknium/OpenHermes-2.5) as the source of our dataset. We select the single-turn dialogue data (612687) from it and append the `length` and `num_tokens`. The content format of these data is as follows:

```plain text
{
	'conversations': [{
			'from': 'human',
			'value': 'Every day...'
		},
		{
			'from': 'gpt',
			'value': "Here's the ..."
		}
	],
	'source': 'airoboros2.2',
	'category': 'orca',
	'skip_prompt_formatting': False
}
```

```shell
python data_process/raw_openhermes_process.py\
    --dataset_path ../datasets/OpenHermes/openhermes2_5.json\
    --output_path ../datasets/single_turn_openhermes.jsonl
```

### Build Target Length Generation Dataset

```shell
python data_process/build_tlg_dataset.py\
    --dataset_path ../datasets/single_turn_openhermes.jsonl\
    --num 2000\
    --random_seed 10\
    --output_path ../datasets/tlg_dataset.jsonl
```

**TLG Dataset:**

```plain text
{
  'id': '0',
	'Instruction': 'How can I generate an AI model that can classify articles of clothing as shorts, skirts, or pants based on their descriptions?',
	'TargetLength': '50'
}
[...]
{
  'id': '1999',
	'Instruction': "I'm currently exploring the use of Slim as a replacement for HAML in my personal project. However, I've noticed that Slim doesn't handle HTML5 data attributes as smoothly as HAML does. I was wondering if anyone has encountered this issue before or if there's a syntax or option in Slim that I haven't come across in their documentation.\n\nIn HAML, I can easily define HTML5 data attributes using nested hashes like this:\n\n`%a{data: {key1: 'val', key2: 'val'}}`\n\nThis would result in:\n\n`<a data-key1='val' data-key2='val'></a>`\n\nIs there an equivalent way to achieve this in Slim?",
	'TargetLength': '300'
}
```

### Run on Target Length Generation Task

prompt:

`<Instruction>The response should have a word count of <TargetLength> words.`

If the target length is `>800`, `>` will be replaced with `more than`.

#### Closed-source Model

```shell
python exp/run_exp_api.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model <MODEL>\
  --output_path ../outputs/tlg/tlg_<MODEL_NAME>.jsonl
  --key <MODEL KEY>
```

<details>
<summary>Shell Commands of Closed-source Models</summary>

**gpt-4-turbo**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model gpt-4-turbo\
  --output_path ../outputs/tlg/tlg_gpt-4-turbo.jsonl
```

**gpt-4o**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model gpt-4o\
  --output_path ../outputs/tlg/tlg_gpt-4o.jsonl
```

**gpt-3.5-turbo**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model gpt-3.5-turbo\
  --output_path ../outputs/tlg/tlg_gpt-3.5-turbo.jsonl
```

**claude-3-haiku**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model claude-3-haiku\
  --output_path ../outputs/tlg/tlg_claude-3-haiku.jsonl
```

**claude-3.5-sonnet**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model claude-3.5-sonnet\
  --output_path ../outputs/tlg/tlg_claude-3.5-sonnet.jsonl
```

</details>

#### Open-source Model

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path <PATH_TO_THE_MODEL>\
  --output_path ../outputs/tlg/tlg_<MODEL_NAME>.jsonl
```

<details>
<summary>Shell Commands of Open-source Models</summary>

**mistralai/Mistral-7B-Instruct-v0.3**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3\
  --output_path ../outputs/tlg/tlg_Mistral-7B-Instruct-v0.3.jsonl
```

**google/gemma-2b-it**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path google/gemma-2b-it\
  --output_path ../outputs/tlg/tlg_gemma-2b-it.jsonl
```

**google/gemma-7b-it**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path google/gemma-7b-it\
  --output_path ../outputs/tlg/tlg_gemma-7b-it.jsonl
```

**meta-llama/Meta-Llama-3-8B-instruct**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path meta-llama/Meta-Llama-3-8B-instruct\
  --output_path ../outputs/tlg/tlg_Meta-Llama-3-8B-instruct.jsonl
```

**meta-llama/Meta-Llama-3-70B-instruct**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path meta-llama/Meta-Llama-3-70B-instruct\
  --output_path ../outputs/tlg/tlg_Meta-Llama-3-70B-instruct.jsonl
```

**internlm/internlm2-chat-7b**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path internlm/internlm2-chat-7b\
  --output_path ../outputs/tlg/tlg_internlm2-chat-7b.jsonl
```

**internlm/internlm2-chat-20b**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path internlm/internlm2-chat-20b\
  --output_path ../outputs/tlg/tlg_internlm2-chat-20b.jsonl
```

**deepseek-ai/deepseek-llm-7b-chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path deepseek-ai/deepseek-llm-7b-chat\
  --output_path ../outputs/tlg/tlg_deepseek-llm-7b-chat.jsonl
```

**deepseek-ai/deepseek-llm-67b-chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path deepseek-ai/deepseek-llm-67b-chat\
  --output_path ../outputs/tlg/tlg_deepseek-llm-67b-chat.jsonl
```

**01-ai/Yi-1.5-6B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path 01-ai/Yi-1.5-6B-Chat\
  --output_path ../outputs/tlg/tlg_Yi-1.5-6B-Chat.jsonl
```

**01-ai/Yi-1.5-9B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path 01-ai/Yi-1.5-9B-Chat\
  --output_path ../outputs/tlg/tlg_Yi-1.5-9B-Chat.jsonl
```

**01-ai/Yi-1.5-34B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path 01-ai/Yi-1.5-34B-Chat\
  --output_path ../outputs/tlg/tlg_Yi-1.5-34B-Chat.jsonl
```

**Qwen/Qwen1.5-7B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path Qwen/Qwen1.5-7B-Chat\
  --output_path ../outputs/tlg/tlg_Qwen1.5-7B-Chat.jsonl
```

**Qwen/Qwen1.5-14B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path Qwen/Qwen1.5-14B-Chat\
  --output_path ../outputs/tlg/tlg_Qwen1.5-14B-Chat.jsonl
```

**Qwen/Qwen1.5-32B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path Qwen/Qwen1.5-32B-Chat\
  --output_path ../outputs/tlg/tlg_Qwen1.5-32B-Chat.jsonl
```

**Qwen/Qwen1.5-72B-Chat**

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/tlg_dataset.jsonl\
  --model_name_or_path Qwen/Qwen1.5-72B-Chat\
  --output_path ../outputs/tlg/tlg_Qwen1.5-72B-Chat.jsonl
```

</details>

### Calculate Scores of Target Length Generation Task

We calculated the `PM` and `FM` scores at different `levels` for each model separately. To provide more detailed scores, we also calculated the `PM` and `FM` scores for each `MLT` individually. The commands are as follows:

Different `Levels`:

```shell
python exp/cal_level_scores.py\
  --dataset_path <PATH_TO_GENERATED_JSONL>
```

Different `MLT`:

```shell
python exp/cal_mlt_scores.py\
  --dataset_path <PATH_TO_GENERATED_JSONL>
```

<details>
<summary>Shell Commands and Results of All Models</summary>

**gpt-4-turbo**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_gpt-4-turbo.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 742   | 160    | 82.26 | 779   | 123    | 86.36 |
| Level:1   | 305   | 351    | 46.49 | 558   | 98     | 85.06 |
| Level:2   | 180   | 262    | 40.72 | 210   | 232    | 47.51 |
| All Level | 1227  | 773    | 61.35 | 1547  | 453    | 77.35 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_gpt-4-turbo.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 195   | 23     | 89.45 | 195   | 23     | 89.45 |
  | 30    | 187   | 28     | 86.98 | 187   | 28     | 86.98 |
  | 50    | 205   | 44     | 82.33 | 205   | 44     | 82.33 |
  | 80    | 155   | 65     | 70.45 | 192   | 28     | 87.27 |
  | Total | 742   | 160    | 82.26 | 779   | 123    | 86.36 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 134   | 62     | 68.37 | 184   | 12     | 93.88 |
  | 300   | 53    | 184    | 22.36 | 199   | 38     | 83.97 |
  | 500   | 118   | 105    | 52.91 | 175   | 48     | 78.48 |
  | Total | 305   | 351    | 46.49 | 558   | 98     | 85.06 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 114   | 116    | 49.57 | 144   | 86     | 62.61 |
  | >800  | 66    | 146    | 31.13 | 66    | 146    | 31.13 |
  | Total | 180   | 262    | 40.72 | 210   | 232    | 47.51 |

**gpt-4o**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_gpt-4o.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 668   | 234    | 74.06 | 713   | 189    | 79.05 |
| Level:1   | 212   | 444    | 32.32 | 455   | 201    | 69.36 |
| Level:2   | 275   | 167    | 62.22 | 318   | 124    | 71.95 |
| All Level | 1155  | 845    | 57.75 | 1486  | 514    | 74.30 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_gpt-4o.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 182   | 36     | 83.49 | 182   | 36     | 83.49 |
  | 30    | 173   | 42     | 80.47 | 173   | 42     | 80.47 |
  | 50    | 177   | 72     | 71.08 | 177   | 72     | 71.08 |
  | 80    | 136   | 84     | 61.82 | 181   | 39     | 82.27 |
  | Total | 668   | 234    | 74.06 | 713   | 189    | 79.05 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 118   | 78     | 60.20 | 174   | 22     | 88.78 |
  | 300   | 37    | 200    | 15.61 | 169   | 68     | 71.31 |
  | 500   | 57    | 166    | 25.56 | 112   | 111    | 50.22 |
  | Total | 212   | 444    | 32.32 | 455   | 201    | 69.36 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 106   | 124    | 46.09 | 149   | 81     | 64.78 |
  | >800  | 169   | 43     | 79.72 | 169   | 43     | 79.72 |
  | Total | 275   | 167    | 62.22 | 318   | 124    | 71.95 |

**gpt-3.5-turbo**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_gpt-3.5-turbo.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 581   | 321    | 64.41 | 630   | 272    | 69.84 |
| Level:1   | 230   | 426    | 35.06 | 497   | 159    | 75.76 |
| Level:2   | 169   | 273    | 38.24 | 203   | 239    | 45.93 |
| All Level | 980   | 1020   | 49.00 | 1330  | 670    | 66.50 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_gpt-3.5-turbo.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 176   | 42     | 80.73 | 176   | 42     | 80.73 |
  | 30    | 155   | 60     | 72.09 | 155   | 60     | 72.09 |
  | 50    | 143   | 106    | 57.43 | 143   | 106    | 57.43 |
  | 80    | 107   | 113    | 48.64 | 156   | 64     | 70.91 |
  | Total | 581   | 321    | 64.41 | 630   | 272    | 69.84 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 106   | 90     | 54.08 | 156   | 40     | 79.59 |
  | 300   | 36    | 201    | 15.19 | 194   | 43     | 81.86 |
  | 500   | 88    | 135    | 39.46 | 147   | 76     | 65.92 |
  | Total | 230   | 426    | 35.06 | 497   | 159    | 75.76 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 82    | 148    | 35.65 | 116   | 114    | 50.43 |
  | >800  | 87    | 125    | 41.04 | 87    | 125    | 41.04 |
  | Total | 169   | 273    | 38.24 | 203   | 239    | 45.93 |

**claude-3-haiku**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_claude-3-haiku.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 435   | 467    | 48.23 | 498   | 404    | 55.21 |
| Level:1   | 232   | 424    | 35.37 | 484   | 172    | 73.78 |
| Level:2   | 195   | 247    | 44.12 | 223   | 219    | 50.45 |
| All Level | 862   | 1138   | 43.10 | 1205  | 795    | 60.25 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_claude-3-haiku.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 151   | 67     | 69.27 | 151   | 67     | 69.27 |
  | 30    | 117   | 98     | 54.42 | 117   | 98     | 54.42 |
  | 50    | 105   | 144    | 42.17 | 105   | 144    | 42.17 |
  | 80    | 62    | 158    | 28.18 | 125   | 95     | 56.82 |
  | Total | 435   | 467    | 48.23 | 498   | 404    | 55.21 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 86    | 110    | 43.88 | 149   | 47     | 76.02 |
  | 300   | 46    | 191    | 19.41 | 180   | 57     | 75.95 |
  | 500   | 100   | 123    | 44.84 | 155   | 68     | 69.51 |
  | Total | 232   | 424    | 35.37 | 484   | 172    | 73.78 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 91    | 139    | 39.57 | 119   | 111    | 51.74 |
  | >800  | 104   | 108    | 49.06 | 104   | 108    | 49.06 |
  | Total | 195   | 247    | 44.12 | 223   | 219    | 50.45 |

**claude-3.5-sonnet**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_claude-3.5-sonnet.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 678   | 224    | 75.17 | 731   | 171    | 81.04 |
| Level:1   | 278   | 378    | 42.38 | 545   | 111    | 83.08 |
| Level:2   | 277   | 165    | 62.67 | 315   | 127    | 71.27 |
| All Level | 1233  | 767    | 61.65 | 1591  | 409    | 79.55 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_claude-3.5-sonnet.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 180   | 38     | 82.57 | 180   | 38     | 82.57 |
  | 30    | 160   | 55     | 74.42 | 160   | 55     | 74.42 |
  | 50    | 188   | 61     | 75.50 | 188   | 61     | 75.50 |
  | 80    | 150   | 70     | 68.18 | 203   | 17     | 92.27 |
  | Total | 678   | 224    | 75.17 | 731   | 171    | 81.04 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 150   | 46     | 76.53 | 191   | 5      | 97.45 |
  | 300   | 57    | 180    | 24.05 | 210   | 27     | 88.61 |
  | 500   | 71    | 152    | 31.84 | 144   | 79     | 64.57 |
  | Total | 278   | 378    | 42.38 | 545   | 111    | 83.08 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 84    | 146    | 36.52 | 122   | 108    | 53.04 |
  | >800  | 193   | 19     | 91.04 | 193   | 19     | 91.04 |
  | Total | 277   | 165    | 62.67 | 315   | 127    | 71.27 |

**mistralai/Mistral-7B-Instruct-v0.3**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Mistral-7B-Instruct-v0.3.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 183   | 719    | 20.29 | 212   | 690    | 23.50 |
| Level:1   | 110   | 546    | 16.77 | 317   | 339    | 48.32 |
| Level:2   | 16    | 426    | 3.62  | 25    | 417    | 5.66  |
| All Level | 309   | 1691   | 15.45 | 554   | 1446   | 27.70 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Mistral-7B-Instruct-v0.3.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 67    | 151    | 30.73 | 67    | 151    | 30.73 |
  | 30    | 40    | 175    | 18.60 | 40    | 175    | 18.60 |
  | 50    | 42    | 207    | 16.87 | 42    | 207    | 16.87 |
  | 80    | 34    | 186    | 15.45 | 63    | 157    | 28.64 |
  | Total | 183   | 719    | 20.29 | 212   | 690    | 23.50 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 35    | 161    | 17.86 | 82    | 114    | 41.84 |
  | 300   | 35    | 202    | 14.77 | 166   | 71     | 70.04 |
  | 500   | 40    | 183    | 17.94 | 69    | 154    | 30.94 |
  | Total | 110   | 546    | 16.77 | 317   | 339    | 48.32 |

- LEVEL2

  | Level | PM_in | PM_out | PM   | FM_in | FM_out | FM   |
  | ----- | ----- | ------ | ---- | ----- | ------ | ---- |
  | 700   | 7     | 223    | 3.04 | 16    | 214    | 6.96 |
  | >800  | 9     | 203    | 4.25 | 9     | 203    | 4.25 |
  | Total | 16    | 426    | 3.62 | 25    | 417    | 5.66 |

**google/gemma-2b-it**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_gemma-2b-it.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 189   | 713    | 20.95 | 209   | 693    | 23.17 |
| Level:1   | 57    | 599    | 8.69  | 159   | 497    | 24.24 |
| Level:2   | 1     | 441    | 0.23  | 1     | 441    | 0.23  |
| All Level | 247   | 1753   | 12.35 | 369   | 1631   | 18.45 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_gemma-2b-it.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 47    | 171    | 21.56 | 47    | 171    | 21.56 |
  | 30    | 65    | 150    | 30.23 | 65    | 150    | 30.23 |
  | 50    | 52    | 197    | 20.88 | 52    | 197    | 20.88 |
  | 80    | 25    | 195    | 11.36 | 45    | 175    | 20.45 |
  | Total | 189   | 713    | 20.95 | 209   | 693    | 23.17 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 34    | 162    | 17.35 | 64    | 132    | 32.65 |
  | 300   | 17    | 220    | 7.17  | 79    | 158    | 33.33 |
  | 500   | 6     | 217    | 2.69  | 16    | 207    | 7.17  |
  | Total | 57    | 599    | 8.69  | 159   | 497    | 24.24 |

- LEVEL2

  | Level | PM_in | PM_out | PM   | FM_in | FM_out | FM   |
  | ----- | ----- | ------ | ---- | ----- | ------ | ---- |
  | 700   | 0     | 230    | 0.00 | 0     | 230    | 0.00 |
  | >800  | 1     | 211    | 0.47 | 1     | 211    | 0.47 |
  | Total | 1     | 441    | 0.23 | 1     | 441    | 0.23 |

**google/gemma-7b-it**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_gemma-7b-it.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 140   | 762    | 15.52 | 170   | 732    | 18.85 |
| Level:1   | 77    | 579    | 11.74 | 235   | 421    | 35.82 |
| Level:2   | 2     | 440    | 0.45  | 2     | 440    | 0.45  |
| All Level | 219   | 1781   | 10.95 | 407   | 1593   | 20.35 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_gemma-7b-it.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 27    | 191    | 12.39 | 27    | 191    | 12.39 |
  | 30    | 39    | 176    | 18.14 | 39    | 176    | 18.14 |
  | 50    | 47    | 202    | 18.88 | 47    | 202    | 18.88 |
  | 80    | 27    | 193    | 12.27 | 57    | 163    | 25.91 |
  | Total | 140   | 762    | 15.52 | 170   | 732    | 18.85 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 37    | 159    | 18.88 | 83    | 113    | 42.35 |
  | 300   | 29    | 208    | 12.24 | 123   | 114    | 51.90 |
  | 500   | 11    | 212    | 4.93  | 29    | 194    | 13.00 |
  | Total | 77    | 579    | 11.74 | 235   | 421    | 35.82 |

- LEVEL2

  | Level | PM_in | PM_out | PM   | FM_in | FM_out | FM   |
  | ----- | ----- | ------ | ---- | ----- | ------ | ---- |
  | 700   | 2     | 228    | 0.87 | 2     | 228    | 0.87 |
  | >800  | 0     | 212    | 0.00 | 0     | 212    | 0.00 |
  | Total | 2     | 440    | 0.45 | 2     | 440    | 0.45 |

**meta-llama/Meta-Llama-3-8B-instruct**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Meta-Llama-3-8B-instruct.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 312   | 590    | 34.59 | 361   | 541    | 40.02 |
| Level:1   | 195   | 461    | 29.73 | 431   | 225    | 65.70 |
| Level:2   | 80    | 362    | 18.10 | 93    | 349    | 21.04 |
| All Level | 587   | 1413   | 29.35 | 885   | 1115   | 44.25 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Meta-Llama-3-8B-instruct.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 99    | 119    | 45.41 | 99    | 119    | 45.41 |
  | 30    | 76    | 139    | 35.35 | 76    | 139    | 35.35 |
  | 50    | 84    | 165    | 33.73 | 84    | 165    | 33.73 |
  | 80    | 53    | 167    | 24.09 | 102   | 118    | 46.36 |
  | Total | 312   | 590    | 34.59 | 361   | 541    | 40.02 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 75    | 121    | 38.27 | 139   | 57     | 70.92 |
  | 300   | 64    | 173    | 27.00 | 187   | 50     | 78.90 |
  | 500   | 56    | 167    | 25.11 | 105   | 118    | 47.09 |
  | Total | 195   | 461    | 29.73 | 431   | 225    | 65.70 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 37    | 193    | 16.09 | 50    | 180    | 21.74 |
  | >800  | 43    | 169    | 20.28 | 43    | 169    | 20.28 |
  | Total | 80    | 362    | 18.10 | 93    | 349    | 21.04 |

**meta-llama/Meta-Llama-3-70B-instruct**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Meta-Llama-3-70B-instruct.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 530   | 372    | 58.76 | 582   | 320    | 64.52 |
| Level:1   | 240   | 416    | 36.59 | 511   | 145    | 77.90 |
| Level:2   | 161   | 281    | 36.43 | 182   | 260    | 41.18 |
| All Level | 931   | 1069   | 46.55 | 1275  | 725    | 63.75 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Meta-Llama-3-70B-instruct.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 132   | 86     | 60.55 | 132   | 86     | 60.55 |
  | 30    | 142   | 73     | 66.05 | 142   | 73     | 66.05 |
  | 50    | 153   | 96     | 61.45 | 153   | 96     | 61.45 |
  | 80    | 103   | 117    | 46.82 | 155   | 65     | 70.45 |
  | Total | 530   | 372    | 58.76 | 582   | 320    | 64.52 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 108   | 88     | 55.10 | 168   | 28     | 85.71 |
  | 300   | 53    | 184    | 22.36 | 210   | 27     | 88.61 |
  | 500   | 79    | 144    | 35.43 | 133   | 90     | 59.64 |
  | Total | 240   | 416    | 36.59 | 511   | 145    | 77.90 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 56    | 174    | 24.35 | 77    | 153    | 33.48 |
  | >800  | 105   | 107    | 49.53 | 105   | 107    | 49.53 |
  | Total | 161   | 281    | 36.43 | 182   | 260    | 41.18 |

**internlm/internlm2-chat-7b**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_internlm2-chat-7b.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 60    | 842    | 6.65  | 65    | 837    | 7.21  |
| Level:1   | 57    | 599    | 8.69  | 180   | 476    | 27.44 |
| Level:2   | 87    | 355    | 19.68 | 99    | 343    | 22.40 |
| All Level | 204   | 1796   | 10.20 | 344   | 1656   | 17.20 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_internlm2-chat-7b.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 39    | 179    | 17.89 | 39    | 179    | 17.89 |
  | 30    | 15    | 200    | 6.98  | 15    | 200    | 6.98  |
  | 50    | 3     | 246    | 1.20  | 3     | 246    | 1.20  |
  | 80    | 3     | 217    | 1.36  | 8     | 212    | 3.64  |
  | Total | 60    | 842    | 6.65  | 65    | 837    | 7.21  |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 18    | 178    | 9.18  | 41    | 155    | 20.92 |
  | 300   | 14    | 223    | 5.91  | 89    | 148    | 37.55 |
  | 500   | 25    | 198    | 11.21 | 50    | 173    | 22.42 |
  | Total | 57    | 599    | 8.69  | 180   | 476    | 27.44 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 43    | 187    | 18.70 | 55    | 175    | 23.91 |
  | >800  | 44    | 168    | 20.75 | 44    | 168    | 20.75 |
  | Total | 87    | 355    | 19.68 | 99    | 343    | 22.40 |

**internlm/internlm2-chat-20b**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_internlm2-chat-20b.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 81    | 821    | 8.98  | 89    | 813    | 9.87  |
| Level:1   | 72    | 584    | 10.98 | 226   | 430    | 34.45 |
| Level:2   | 77    | 365    | 17.42 | 89    | 353    | 20.14 |
| All Level | 230   | 1770   | 11.50 | 404   | 1596   | 20.20 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_internlm2-chat-20b.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 45    | 173    | 20.64 | 45    | 173    | 20.64 |
  | 30    | 19    | 196    | 8.84  | 19    | 196    | 8.84  |
  | 50    | 7     | 242    | 2.81  | 7     | 242    | 2.81  |
  | 80    | 10    | 210    | 4.55  | 18    | 202    | 8.18  |
  | Total | 81    | 821    | 8.98  | 89    | 813    | 9.87  |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 19    | 177    | 9.69  | 45    | 151    | 22.96 |
  | 300   | 22    | 215    | 9.28  | 109   | 128    | 45.99 |
  | 500   | 31    | 192    | 13.90 | 72    | 151    | 32.29 |
  | Total | 72    | 584    | 10.98 | 226   | 430    | 34.45 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 40    | 190    | 17.39 | 52    | 178    | 22.61 |
  | >800  | 37    | 175    | 17.45 | 37    | 175    | 17.45 |
  | Total | 77    | 365    | 17.42 | 89    | 353    | 20.14 |

**deepseek-ai/deepseek-llm-7b-chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_deepseek-llm-7b-chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 254   | 648    | 28.16 | 283   | 619    | 31.37 |
| Level:1   | 116   | 540    | 17.68 | 291   | 365    | 44.36 |
| Level:2   | 48    | 394    | 10.86 | 58    | 384    | 13.12 |
| All Level | 418   | 1582   | 20.90 | 632   | 1368   | 31.60 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_deepseek-llm-7b-chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 127   | 91     | 58.26 | 127   | 91     | 58.26 |
  | 30    | 54    | 161    | 25.12 | 54    | 161    | 25.12 |
  | 50    | 44    | 205    | 17.67 | 44    | 205    | 17.67 |
  | 80    | 29    | 191    | 13.18 | 58    | 162    | 26.36 |
  | Total | 254   | 648    | 28.16 | 283   | 619    | 31.37 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 30    | 166    | 15.31 | 73    | 123    | 37.24 |
  | 300   | 43    | 194    | 18.14 | 144   | 93     | 60.76 |
  | 500   | 43    | 180    | 19.28 | 74    | 149    | 33.18 |
  | Total | 116   | 540    | 17.68 | 291   | 365    | 44.36 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 21    | 209    | 9.13  | 31    | 199    | 13.48 |
  | >800  | 27    | 185    | 12.74 | 27    | 185    | 12.74 |
  | Total | 48    | 394    | 10.86 | 58    | 384    | 13.12 |

**deepseek-ai/deepseek-llm-67b-chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_deepseek-llm-67b-chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 243   | 659    | 26.94 | 273   | 629    | 30.27 |
| Level:1   | 112   | 544    | 17.07 | 325   | 331    | 49.54 |
| Level:2   | 42    | 400    | 9.50  | 53    | 389    | 11.99 |
| All Level | 397   | 1603   | 19.85 | 651   | 1349   | 32.55 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_deepseek-llm-67b-chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 102   | 116    | 46.79 | 102   | 116    | 46.79 |
  | 30    | 44    | 171    | 20.47 | 44    | 171    | 20.47 |
  | 50    | 55    | 194    | 22.09 | 55    | 194    | 22.09 |
  | 80    | 42    | 178    | 19.09 | 72    | 148    | 32.73 |
  | Total | 243   | 659    | 26.94 | 273   | 629    | 30.27 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 18    | 178    | 9.18  | 68    | 128    | 34.69 |
  | 300   | 47    | 190    | 19.83 | 170   | 67     | 71.73 |
  | 500   | 47    | 176    | 21.08 | 87    | 136    | 39.01 |
  | Total | 112   | 544    | 17.07 | 325   | 331    | 49.54 |

- LEVEL2

  | Level | PM_in | PM_out | PM   | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ---- | ----- | ------ | ----- |
  | 700   | 21    | 209    | 9.13 | 32    | 198    | 13.91 |
  | >800  | 21    | 191    | 9.91 | 21    | 191    | 9.91  |
  | Total | 42    | 400    | 9.50 | 53    | 389    | 11.99 |

**01-ai/Yi-1.5-6B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Yi-1.5-6B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 212   | 690    | 23.50 | 233   | 669    | 25.83 |
| Level:1   | 108   | 548    | 16.46 | 320   | 336    | 48.78 |
| Level:2   | 80    | 362    | 18.10 | 90    | 352    | 20.36 |
| All Level | 400   | 1600   | 20.00 | 643   | 1357   | 32.15 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Yi-1.5-6B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 87    | 131    | 39.91 | 87    | 131    | 39.91 |
  | 30    | 51    | 164    | 23.72 | 51    | 164    | 23.72 |
  | 50    | 50    | 199    | 20.08 | 50    | 199    | 20.08 |
  | 80    | 24    | 196    | 10.91 | 45    | 175    | 20.45 |
  | Total | 212   | 690    | 23.50 | 233   | 669    | 25.83 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 37    | 159    | 18.88 | 92    | 104    | 46.94 |
  | 300   | 30    | 207    | 12.66 | 148   | 89     | 62.45 |
  | 500   | 41    | 182    | 18.39 | 80    | 143    | 35.87 |
  | Total | 108   | 548    | 16.46 | 320   | 336    | 48.78 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 29    | 201    | 12.61 | 39    | 191    | 16.96 |
  | >800  | 51    | 161    | 24.06 | 51    | 161    | 24.06 |
  | Total | 80    | 362    | 18.10 | 90    | 352    | 20.36 |

**01-ai/Yi-1.5-9B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Yi-1.5-9B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 228   | 674    | 25.28 | 263   | 639    | 29.16 |
| Level:1   | 114   | 542    | 17.38 | 291   | 365    | 44.36 |
| Level:2   | 108   | 334    | 24.43 | 130   | 312    | 29.41 |
| All Level | 450   | 1550   | 22.50 | 684   | 1316   | 34.20 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Yi-1.5-9B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 104   | 114    | 47.71 | 104   | 114    | 47.71 |
  | 30    | 51    | 164    | 23.72 | 51    | 164    | 23.72 |
  | 50    | 43    | 206    | 17.27 | 43    | 206    | 17.27 |
  | 80    | 30    | 190    | 13.64 | 65    | 155    | 29.55 |
  | Total | 228   | 674    | 25.28 | 263   | 639    | 29.16 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 25    | 171    | 12.76 | 65    | 131    | 33.16 |
  | 300   | 30    | 207    | 12.66 | 127   | 110    | 53.59 |
  | 500   | 59    | 164    | 26.46 | 99    | 124    | 44.39 |
  | Total | 114   | 542    | 17.38 | 291   | 365    | 44.36 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 51    | 179    | 22.17 | 73    | 157    | 31.74 |
  | >800  | 57    | 155    | 26.89 | 57    | 155    | 26.89 |
  | Total | 108   | 334    | 24.43 | 130   | 312    | 29.41 |

**01-ai/Yi-1.5-34B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Yi-1.5-34B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 260   | 642    | 28.82 | 303   | 599    | 33.59 |
| Level:1   | 171   | 485    | 26.07 | 429   | 227    | 65.40 |
| Level:2   | 94    | 348    | 21.27 | 114   | 328    | 25.79 |
| All Level | 525   | 1475   | 26.25 | 846   | 1154   | 42.30 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Yi-1.5-34B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 99    | 119    | 45.41 | 99    | 119    | 45.41 |
  | 30    | 59    | 156    | 27.44 | 59    | 156    | 27.44 |
  | 50    | 51    | 198    | 20.48 | 51    | 198    | 20.48 |
  | 80    | 51    | 169    | 23.18 | 94    | 126    | 42.73 |
  | Total | 260   | 642    | 28.82 | 303   | 599    | 33.59 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 50    | 146    | 25.51 | 115   | 81     | 58.67 |
  | 300   | 57    | 180    | 24.05 | 186   | 51     | 78.48 |
  | 500   | 64    | 159    | 28.70 | 128   | 95     | 57.40 |
  | Total | 171   | 485    | 26.07 | 429   | 227    | 65.40 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 51    | 179    | 22.17 | 71    | 159    | 30.87 |
  | >800  | 43    | 169    | 20.28 | 43    | 169    | 20.28 |
  | Total | 94    | 348    | 21.27 | 114   | 328    | 25.79 |

**Qwen/Qwen1.5-7B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-7B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 219   | 683    | 24.28 | 247   | 655    | 27.38 |
| Level:1   | 94    | 562    | 14.33 | 303   | 353    | 46.19 |
| Level:2   | 40    | 402    | 9.05  | 53    | 389    | 11.99 |
| All Level | 353   | 1647   | 17.65 | 603   | 1397   | 30.15 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-7B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 68    | 150    | 31.19 | 68    | 150    | 31.19 |
  | 30    | 55    | 160    | 25.58 | 55    | 160    | 25.58 |
  | 50    | 57    | 192    | 22.89 | 57    | 192    | 22.89 |
  | 80    | 39    | 181    | 17.73 | 67    | 153    | 30.45 |
  | Total | 219   | 683    | 24.28 | 247   | 655    | 27.38 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 19    | 177    | 9.69  | 58    | 138    | 29.59 |
  | 300   | 17    | 220    | 7.17  | 146   | 91     | 61.60 |
  | 500   | 58    | 165    | 26.01 | 99    | 124    | 44.39 |
  | Total | 94    | 562    | 14.33 | 303   | 353    | 46.19 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 28    | 202    | 12.17 | 41    | 189    | 17.83 |
  | >800  | 12    | 200    | 5.66  | 12    | 200    | 5.66  |
  | Total | 40    | 402    | 9.05  | 53    | 389    | 11.99 |

**Qwen/Qwen1.5-14B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-14B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 255   | 647    | 28.27 | 284   | 618    | 31.49 |
| Level:1   | 121   | 535    | 18.45 | 288   | 368    | 43.90 |
| Level:2   | 49    | 393    | 11.09 | 63    | 379    | 14.25 |
| All Level | 425   | 1575   | 21.25 | 635   | 1365   | 31.75 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-14B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 100   | 118    | 45.87 | 100   | 118    | 45.87 |
  | 30    | 62    | 153    | 28.84 | 62    | 153    | 28.84 |
  | 50    | 66    | 183    | 26.51 | 66    | 183    | 26.51 |
  | 80    | 27    | 193    | 12.27 | 56    | 164    | 25.45 |
  | Total | 255   | 647    | 28.27 | 284   | 618    | 31.49 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 11    | 185    | 5.61  | 33    | 163    | 16.84 |
  | 300   | 26    | 211    | 10.97 | 133   | 104    | 56.12 |
  | 500   | 84    | 139    | 37.67 | 122   | 101    | 54.71 |
  | Total | 121   | 535    | 18.45 | 288   | 368    | 43.90 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 35    | 195    | 15.22 | 49    | 181    | 21.30 |
  | >800  | 14    | 198    | 6.60  | 14    | 198    | 6.60  |
  | Total | 49    | 393    | 11.09 | 63    | 379    | 14.25 |

  **Qwen/Qwen1.5-32B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-32B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 294   | 608    | 32.59 | 327   | 575    | 36.25 |
| Level:1   | 146   | 510    | 22.26 | 324   | 332    | 49.39 |
| Level:2   | 95    | 347    | 21.49 | 112   | 330    | 25.34 |
| All Level | 535   | 1465   | 26.75 | 763   | 1237   | 38.15 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-32B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 102   | 116    | 46.79 | 102   | 116    | 46.79 |
  | 30    | 73    | 142    | 33.95 | 73    | 142    | 33.95 |
  | 50    | 73    | 176    | 29.32 | 73    | 176    | 29.32 |
  | 80    | 46    | 174    | 20.91 | 79    | 141    | 35.91 |
  | Total | 294   | 608    | 32.59 | 327   | 575    | 36.25 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 41    | 155    | 20.92 | 85    | 111    | 43.37 |
  | 300   | 35    | 202    | 14.77 | 127   | 110    | 53.59 |
  | 500   | 70    | 153    | 31.39 | 112   | 111    | 50.22 |
  | Total | 146   | 510    | 22.26 | 324   | 332    | 49.39 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 55    | 175    | 23.91 | 72    | 158    | 31.30 |
  | >800  | 40    | 172    | 18.87 | 40    | 172    | 18.87 |
  | Total | 95    | 347    | 21.49 | 112   | 330    | 25.34 |

**Qwen/Qwen1.5-72B-Chat**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_Qwen1.5-72B-Chat.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 321   | 581    | 35.59 | 358   | 544    | 39.69 |
| Level:1   | 120   | 536    | 18.29 | 326   | 330    | 49.70 |
| Level:2   | 17    | 425    | 3.85  | 27    | 415    | 6.11  |
| All Level | 458   | 1542   | 22.90 | 711   | 1289   | 35.55 |

```shell
python exp/cal_mlt_scores.py\
    --dataset_path ../outputs/tlg/tlg_Qwen1.5-72B-Chat.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 86    | 132    | 39.45 | 86    | 132    | 39.45 |
  | 30    | 90    | 125    | 41.86 | 90    | 125    | 41.86 |
  | 50    | 81    | 168    | 32.53 | 81    | 168    | 32.53 |
  | 80    | 64    | 156    | 29.09 | 101   | 119    | 45.91 |
  | Total | 321   | 581    | 35.59 | 358   | 544    | 39.69 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 26    | 170    | 13.27 | 69    | 127    | 35.20 |
  | 300   | 30    | 207    | 12.66 | 154   | 83     | 64.98 |
  | 500   | 64    | 159    | 28.70 | 103   | 120    | 46.19 |
  | Total | 120   | 536    | 18.29 | 326   | 330    | 49.70 |

- LEVEL2

  | Level | PM_in | PM_out | PM   | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ---- | ----- | ------ | ----- |
  | 700   | 14    | 216    | 6.09 | 24    | 206    | 10.43 |
  | >800  | 3     | 209    | 1.42 | 3     | 209    | 1.42  |
  | Total | 17    | 425    | 3.85 | 27    | 415    | 6.11  |

</details>

## Ruler

### Build Ruler Training Dataset

We build Ruler training dataset from OpenHermes, LongForm and Eli5.

```shell
python data_process/build_training_dataset.py\
    --dataset_path ../datasets/single_turn_openhermes.jsonl\
    --longform_dir ../datasets/LongForm/data\
    --num 2000\
    --random_seed 10\
    --output_path ../datasets/ruler_training_dataset.jsonl
```

### Finetune the Models

We have provided the [finetuning scripts](./scripts/) for the models mentioned in the paper, see in `<MODEL>/ruler.sh`.

If you wish to fine-tune other models or customize a model, you can modify the relevant parameters as needed. Additionally, we have included annotations in the script for guidance. Also you need add the template of the model in `src/utils/templates.py`.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3

find_free_port() {
    while :
    do
        PORT=$(( ( RANDOM % 64512 ) + 1024 ))
        (echo >/dev/tcp/localhost/$PORT) >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo $PORT
            return
        fi
    done
}

export MASTER_PORT=$(find_free_port)

LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=3
VANILLA=False

MODEL_NAME_OR_PATH=<MODEL_NAME_OR_PATH>
echo "Finetune from: ${MODEL_NAME_OR_PATH}"
MODEL=${MODEL_NAME_OR_PATH##*/}

TEMPLATE=custom
echo "Finetune data template: ${TEMPLATE}"

DATA_PATH=../datasets/ruler_training_dataset.jsonl
echo "Finetune data path: ${DATA_PATH}"

MODEL_MAX_LENGTH=2048
echo "Model max length: ${MODEL_MAX_LENGTH}"

BATCH_SIZE=4
echo "Per device train batch size: ${BATCH_SIZE}"

GRAD_ACCUM=8
echo "Gradient accumulation steps: ${GRAD_ACCUM}"

OUTPUT_DIR="../outputs/checkpoints/ruler_${MODEL}_bs_${BATCH_SIZE}_ga_${GRAD_ACCUM}_lr_${LEARNING_RATE}_eps_${NUM_TRAIN_EPOCHS}"
LOG_DIR=../logs

deepspeed --master_port=$MASTER_PORT finetuning/finetune.py \
  --vanilla $VANILLA \
  --deepspeed ../configs/ds_config_zero3.json \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --template $TEMPLATE\
  --model_max_length $MODEL_MAX_LENGTH \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --bf16 True \
  --tf32 True \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --gradient_checkpointing True \
  --lr_scheduler_type cosine \
  --learning_rate ${LEARNING_RATE} \
  --warmup_ratio 0.05 \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_steps 5 \
  2>&1 | tee ${LOG_DIR}/output_ruler_${MODEL}.log
```

### TLG Results

<details>
<summary>Shell Commands and Results of All Models</summary>

**Ruler + Mistral-7B-v0.3**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Mistral-7B-v0.3.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 633   | 269    | 70.18 | 677   | 225    | 75.06 |
| Level:1   | 233   | 423    | 35.52 | 445   | 211    | 67.84 |
| Level:2   | 149   | 293    | 33.71 | 161   | 281    | 36.43 |
| All Level | 1015  | 985    | 50.75 | 1283  | 717    | 64.15 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Mistral-7B-v0.3.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 164   | 54     | 75.23 | 164   | 54     | 75.23 |
  | 30    | 145   | 70     | 67.44 | 145   | 70     | 67.44 |
  | 50    | 172   | 77     | 69.08 | 172   | 77     | 69.08 |
  | 80    | 152   | 68     | 69.09 | 196   | 24     | 89.09 |
  | Total | 633   | 269    | 70.18 | 677   | 225    | 75.06 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 107   | 89     | 54.59 | 166   | 30     | 84.69 |
  | 300   | 61    | 176    | 25.74 | 169   | 68     | 71.31 |
  | 500   | 65    | 158    | 29.15 | 110   | 113    | 49.33 |
  | Total | 233   | 423    | 35.52 | 445   | 211    | 67.84 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 31    | 199    | 13.48 | 43    | 187    | 18.70 |
  | >800  | 118   | 94     | 55.66 | 118   | 94     | 55.66 |
  | Total | 149   | 293    | 33.71 | 161   | 281    | 36.43 |

**Ruler + gemma-7b**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_gemma-7b.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 537   | 365    | 59.53 | 579   | 323    | 64.19 |
| Level:1   | 258   | 398    | 39.33 | 447   | 209    | 68.14 |
| Level:2   | 112   | 330    | 25.34 | 123   | 319    | 27.83 |
| All Level | 907   | 1093   | 45.35 | 1149  | 851    | 57.45 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_gemma-7b.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 121   | 97     | 55.50 | 121   | 97     | 55.50 |
  | 30    | 142   | 73     | 66.05 | 142   | 73     | 66.05 |
  | 50    | 144   | 105    | 57.83 | 144   | 105    | 57.83 |
  | 80    | 130   | 90     | 59.09 | 172   | 48     | 78.18 |
  | Total | 537   | 365    | 59.53 | 579   | 323    | 64.19 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 113   | 83     | 57.65 | 168   | 28     | 85.71 |
  | 300   | 71    | 166    | 29.96 | 174   | 63     | 73.42 |
  | 500   | 74    | 149    | 33.18 | 105   | 118    | 47.09 |
  | Total | 258   | 398    | 39.33 | 447   | 209    | 68.14 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 16    | 214    | 6.96  | 27    | 203    | 11.74 |
  | >800  | 96    | 116    | 45.28 | 96    | 116    | 45.28 |
  | Total | 112   | 330    | 25.34 | 123   | 319    | 27.83 |

**Ruler + Meta-Llama-3-8B**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Meta-Llama-3-8B.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 697   | 205    | 77.27 | 728   | 174    | 80.71 |
| Level:1   | 333   | 323    | 50.76 | 550   | 106    | 83.84 |
| Level:2   | 85    | 357    | 19.23 | 101   | 341    | 22.85 |
| All Level | 1115  | 885    | 55.75 | 1379  | 621    | 68.95 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Meta-Llama-3-8B.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 179   | 39     | 82.11 | 179   | 39     | 82.11 |
  | 30    | 171   | 44     | 79.53 | 171   | 44     | 79.53 |
  | 50    | 188   | 61     | 75.50 | 188   | 61     | 75.50 |
  | 80    | 159   | 61     | 72.27 | 190   | 30     | 86.36 |
  | Total | 697   | 205    | 77.27 | 728   | 174    | 80.71 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 142   | 54     | 72.45 | 183   | 13     | 93.37 |
  | 300   | 113   | 124    | 47.68 | 221   | 16     | 93.25 |
  | 500   | 78    | 145    | 34.98 | 146   | 77     | 65.47 |
  | Total | 333   | 323    | 50.76 | 550   | 106    | 83.84 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 16    | 214    | 6.96  | 32    | 198    | 13.91 |
  | >800  | 69    | 143    | 32.55 | 69    | 143    | 32.55 |
  | Total | 85    | 357    | 19.23 | 101   | 341    | 22.85 |

**Ruler + deepseek-llm-7b-base**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_deepseek-llm-7b-base.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 615   | 287    | 68.18 | 663   | 239    | 73.50 |
| Level:1   | 204   | 452    | 31.10 | 452   | 204    | 68.90 |
| Level:2   | 51    | 391    | 11.54 | 52    | 390    | 11.76 |
| All Level | 870   | 1130   | 43.50 | 1167  | 833    | 58.35 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_deepseek-llm-7b-base.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 151   | 67     | 69.27 | 151   | 67     | 69.27 |
  | 30    | 161   | 54     | 74.88 | 161   | 54     | 74.88 |
  | 50    | 165   | 84     | 66.27 | 165   | 84     | 66.27 |
  | 80    | 138   | 82     | 62.73 | 186   | 34     | 84.55 |
  | Total | 615   | 287    | 68.18 | 663   | 239    | 73.50 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 110   | 86     | 56.12 | 180   | 16     | 91.84 |
  | 300   | 67    | 170    | 28.27 | 207   | 30     | 87.34 |
  | 500   | 27    | 196    | 12.11 | 65    | 158    | 29.15 |
  | Total | 204   | 452    | 31.10 | 452   | 204    | 68.90 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 12    | 218    | 5.22  | 13    | 217    | 5.65  |
  | >800  | 39    | 173    | 18.40 | 39    | 173    | 18.40 |
  | Total | 51    | 391    | 11.54 | 52    | 390    | 11.76 |

**Ruler + Yi-1.5-6B**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Yi-1.5-6B.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 605   | 297    | 67.07 | 651   | 251    | 72.17 |
| Level:1   | 265   | 391    | 40.40 | 504   | 152    | 76.83 |
| Level:2   | 85    | 357    | 19.23 | 93    | 349    | 21.04 |
| All Level | 955   | 1045   | 47.75 | 1248  | 752    | 62.40 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Yi-1.5-6B.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 163   | 55     | 74.77 | 163   | 55     | 74.77 |
  | 30    | 153   | 62     | 71.16 | 153   | 62     | 71.16 |
  | 50    | 155   | 94     | 62.25 | 155   | 94     | 62.25 |
  | 80    | 134   | 86     | 60.91 | 180   | 40     | 81.82 |
  | Total | 605   | 297    | 67.07 | 651   | 251    | 72.17 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 118   | 78     | 60.20 | 174   | 22     | 88.78 |
  | 300   | 75    | 162    | 31.65 | 212   | 25     | 89.45 |
  | 500   | 72    | 151    | 32.29 | 118   | 105    | 52.91 |
  | Total | 265   | 391    | 40.40 | 504   | 152    | 76.83 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 24    | 206    | 10.43 | 32    | 198    | 13.91 |
  | >800  | 61    | 151    | 28.77 | 61    | 151    | 28.77 |
  | Total | 85    | 357    | 19.23 | 93    | 349    | 21.04 |

**Ruler + Qwen1.5-7B**

```shell
python exp/cal_level_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Qwen1.5-7B.jsonl
```

| Level     | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
| --------- | ----- | ------ | ----- | ----- | ------ | ----- |
| Level:0   | 533   | 369    | 59.09 | 581   | 321    | 64.41 |
| Level:1   | 196   | 460    | 29.88 | 402   | 254    | 61.28 |
| Level:2   | 51    | 391    | 11.54 | 63    | 379    | 14.25 |
| All Level | 780   | 1220   | 39.00 | 1046  | 954    | 52.30 |

```shell
python exp/cal_mlt_scores.py\
  --dataset_path ../outputs/tlg/tlg_ruler_Qwen1.5-7B.jsonl
```

- LEVEL0

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 10    | 162   | 56     | 74.31 | 162   | 56     | 74.31 |
  | 30    | 139   | 76     | 64.65 | 139   | 76     | 64.65 |
  | 50    | 124   | 125    | 49.80 | 124   | 125    | 49.80 |
  | 80    | 108   | 112    | 49.09 | 156   | 64     | 70.91 |
  | Total | 533   | 369    | 59.09 | 581   | 321    | 64.41 |

- LEVEL1

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 150   | 108   | 88     | 55.10 | 165   | 31     | 84.18 |
  | 300   | 65    | 172    | 27.43 | 180   | 57     | 75.95 |
  | 500   | 23    | 200    | 10.31 | 57    | 166    | 25.56 |
  | Total | 196   | 460    | 29.88 | 402   | 254    | 61.28 |

- LEVEL2

  | Level | PM_in | PM_out | PM    | FM_in | FM_out | FM    |
  | ----- | ----- | ------ | ----- | ----- | ------ | ----- |
  | 700   | 14    | 216    | 6.09  | 26    | 204    | 11.30 |
  | >800  | 37    | 175    | 17.45 | 37    | 175    | 17.45 |
  | Total | 51    | 391    | 11.54 | 63    | 379    | 14.25 |

</details>

## Multi MLT Generation Experiment

### Build Dataset

You need to download [arena-hard-auto dataset](https://github.com/lm-sys/arena-hard-auto/blob/main/data/arena-hard-v0.1/question.jsonl) and put it in the `datasets` folder.

```shell
mv ../datasets/question.jsonl ../datasets/arena_question.jsonl

python data_process/build_arena_dataset.py\
  --dataset_path ../datasets/arena_question.jsonl\
  --num 200\
  --random_seed 10\
  --output_path ../datasets/multi_mlt.jsonl
```

### Run the Experiment

```shell
python exp/run_exp.py\
  --dataset_path ../data/multi_mlt.jsonl\
  --model_name_or_path <MODEL_NAME_OR_PATH>\
  --gpus 1\
  --template <TEMPLATE>\
  --output_path ../outputs/multi_mlt/mmlt_<MDOEL_NAME>.jsonl
```

### Calculate the Metrics

```shell
python exp/analysis_mmlt.py\
  --dataset_path ../outputs/multi_mlt/mmlt_<MDOEL_NAME>.jsonl
```

| Model                    | 10   | 30   | 50   | 80   | 150  | 300  | 500  | 700  | >800 | Avg FM |
| ------------------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ |
| Mistral-7B-Instruct-v0.3 | 0.5  | 0.0  | 0.5  | 2.0  | 18.5 | 50.5 | 20.5 | 3.0  | 2.5  | 10.89  |
| Mistral-7B-v0.3R         | 72.5 | 68.0 | 65.5 | 76.5 | 76.0 | 63.0 | 28.0 | 24.0 | 64.5 | 59.78  |
| gemma-7b-it              | 13.0 | 17.0 | 15.5 | 26.0 | 54.5 | 76.5 | 17.5 | 0.0  | 0.0  | 24.44  |
| gemma-7bR                | 58.0 | 63.5 | 61.0 | 69.5 | 72.5 | 64.0 | 42.0 | 17.0 | 67.0 | 57.17  |
| Llama-3-8B-Instruct      | 23.5 | 18.0 | 12.5 | 28.0 | 50.5 | 76.5 | 57.0 | 25.5 | 30.5 | 35.78  |
| Llama-3-8BR              | 84.0 | 84.0 | 73.0 | 80.0 | 87.5 | 89.5 | 71.0 | 14.5 | 36.5 | 68.89  |
| deepseek-llm-7b-chat     | 36.5 | 16.0 | 12.5 | 17.5 | 23.5 | 60.5 | 36.5 | 16.0 | 22.5 | 26.83  |
| deepseek-llm-7bR         | 64.0 | 70.0 | 62.5 | 73.0 | 82.0 | 86.5 | 27.0 | 17.0 | 40.5 | 58.06  |
| Yi-1.5-6B-Chat           | 26.5 | 16.5 | 14.5 | 14.5 | 18.5 | 42.5 | 35.0 | 33.5 | 28.5 | 25.56  |
| Yi-1.5-6BR               | 80.5 | 66.0 | 67.0 | 77.0 | 83.5 | 83.5 | 56.0 | 22.0 | 39.5 | 63.89  |
| Qwen1.5-7B-Chat          | 13.5 | 17.0 | 9.5  | 16.0 | 6.5  | 51.0 | 57.5 | 22.5 | 4.5  | 22.00  |
| Qwen1.5-7BR              | 69.0 | 61.0 | 46.5 | 68.5 | 81.0 | 80.5 | 38.5 | 16.5 | 36.5 | 55.33  |

## Self-generated MLT Experiment

### Build Dataset

You also need to download [arena-hard-auto dataset](https://github.com/lm-sys/arena-hard-auto/blob/main/data/arena-hard-v0.1/question.jsonl) and put it in the `datasets` folder.

```shell
mv ../datasets/question.jsonl ../datasets/arena_question.jsonl

python data_process/build_arena_dataset.py\
  --dataset_path ../datasets/arena_question.jsonl\
  --output_path ../datasets/self_generated_mlt.jsonl
```

### Run the Experiment

```shell
python exp/run_exp.py\
  --dataset_path ../datasets/self_generated_mlt.jsonl\
  --model_name_or_path <MODEL_NAME_OR_PATH>\
  --gpus 1\
  --template custom\
  --output_path ../outputs/self_generated_mlt/sgm_<MDOEL_NAME>.jsonl
```

### Calculate the Metrics

```shell
python exp/analysis_sgm.py\
  --dataset_path ../outputs/self_generated_mlt/sgm_<MDOEL_NAME>.jsonl
```

| Model            | 10  | 30  | 50  | 80  | 150 | 300 | 500 | 700 | >800 | FM    | Avg |
| ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----- | --- |
| Mistral-7B-v0.3R | 19  | 87  | 22  | 1   | 208 | 143 | 0   | 0   | 20   | 73.40 | 279 |
| gemma-7bR        | 12  | 133 | 5   | 9   | 200 | 114 | 0   | 0   | 27   | 69.00 | 347 |
| Llama-3-8BR      | 7   | 50  | 23  | 25  | 240 | 141 | 1   | 0   | 13   | 88.40 | 215 |
| deepseek-llm-7bR | 12  | 41  | 18  | 130 | 149 | 147 | 0   | 0   | 3    | 84.40 | 187 |
| Yi-1.5-6BR       | 10  | 19  | 62  | 19  | 233 | 136 | 0   | 0   | 21   | 81.40 | 236 |
| Qwen1.5-7BR      | 9   | 15  | 39  | 81  | 189 | 151 | 1   | 0   | 15   | 81.60 | 245 |

## Overall Performance

We use `lm_eval` to evaluate models on other tasks (`ARC`, `Hellaswag`, `TruthfulQA`, `MMLU`,`Winogrande`, `GSM8K`).

### Vanilla SFT

Use vanilla method to train the models. See each script in `vanilla.sh`

```shell
#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

export MASTER_PORT=$(echo $METIS_WORKER_0_PORT | cut -d',' -f1)

LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=3
VANILLA=True

MODEL_NAME_OR_PATH=<MODEL_NAME_OR_PATH>
echo "Finetune from: ${MODEL_NAME_OR_PATH}"
MODEL=${MODEL_NAME_OR_PATH##*/}

TEMPLATE=custom
echo "Finetune data template: ${TEMPLATE}"

DATA_PATH=../datasets/ruler_training_dataset.jsonl
echo "Finetune data path: ${DATA_PATH}"

MODEL_MAX_LENGTH=2048
echo "Model max length: ${MODEL_MAX_LENGTH}"

BATCH_SIZE=4
echo "Per device train batch size: ${BATCH_SIZE}"

GRAD_ACCUM=8
echo "Gradient accumulation steps: ${GRAD_ACCUM}"

OUTPUT_DIR="../outputs/checkpoints/vanilla_${MODEL}_bs_${BATCH_SIZE}_ga_${GRAD_ACCUM}_lr_${LEARNING_RATE}_eps_${NUM_TRAIN_EPOCHS}"
LOG_DIR=../logs

deepspeed finetuning/finetune.py \
    --vanilla $VANILLA \
    --deepspeed ../configs/ds_config_zero3.json \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --template $TEMPLATE\
    --model_max_length $MODEL_MAX_LENGTH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.05 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --logging_steps 5 \
    2>&1 | tee ${LOG_DIR}/output_vanilla_${MODEL}.log
```

### Overall Performance Test

```shell
set -ex
export NUMEXPR_MAX_THREADS=128

MODEL=vllm
MODEL_NAME=<MODEL_NAME>
MODEL_NAME_OR_PATH=<MODEL_NAME_OR_PATH>
OUTPUT_PATH=../outputs/overall_performance/${MODEL_NAME}
TOKENIZER_MODE=auto
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=0.8

mkdir -p $OUTPUT_PATH

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks ai2_arc \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_ai2_arc \
    --batch_size 1 \
    --num_fewshot 25 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_ai2_arc.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks hellaswag \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_hellaswag \
    --batch_size 1 \
    --num_fewshot 10 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_hellaswag.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks truthfulqa \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_truthfulqa \
    --batch_size 1 \
    --num_fewshot 0 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_truthfulqa.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks mmlu \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_mmlu \
    --batch_size 1 \
    --num_fewshot 5 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_mmlu.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks winogrande \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_winogrande \
    --batch_size 1 \
    --num_fewshot 5 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_winogrande.log

lm_eval --model $MODEL \
    --model_args pretrained=${MODEL_NAME_OR_PATH},trust_remote_code=True,tokenizer_mode=${TOKENIZER_MODE},tensor_parallel_size=${NUM_GPUS},dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    --tasks gsm8k \
    --device cuda \
    --output_path ${OUTPUT_PATH}/${MODEL}_eval_gsm8k \
    --batch_size 1 \
    --num_fewshot 5 \
    --write_out \
    2>&1 | tee ${OUTPUT_PATH}/${MODEL}_eval_gsm8k.log
```

| Model                | Type    | ARC(chanllenge/easy) | HellaSwag | TruthfulQA | MMLU  | Wniograde | GSM8K |
| -------------------- | ------- | -------------------- | --------- | ---------- | ----- | --------- | ----- |
| Mistral-7B-v0.3      | vanilla | 38.23/67.76          | 48.57     | 46.02      | 34.94 | 62.04     | 26.46 |
| -                    | Ruler   | 37.97/67.85          | 47.83     | 47.12      | 37.88 | 62.83     | 27.52 |
| gemma-7b             | vanilla | 35.75/65.66          | 45.95     | 41.13      | 32.44 | 57.14     | 23.58 |
| -                    | Ruler   | 38.99/67.47          | 45.40     | 45.65      | 31.67 | 60.30     | 25.93 |
| Meta-Llama-3-8B      | vanilla | 48.63/77.48          | 58.89     | 51.41      | 50.91 | 71.74     | 44.96 |
| -                    | Ruler   | 49.23/77.99          | 59.12     | 51.90      | 50.16 | 71.19     | 46.63 |
| deepseek-llm-7b-base | vanilla | 50.94/79.92          | 61.48     | 39.90      | 48.65 | 72.93     | 38.89 |
| -                    | Ruler   | 51.37/79.55          | 61.31     | 38.43      | 48.81 | 72.77     | 37.15 |
| Yi-1.5-6B            | vanilla | 51.62/79.25          | 58.79     | 55.32      | 54.68 | 68.51     | 52.01 |
| -                    | Ruler   | 51.28/79.46          | 58.41     | 49.94      | 55.13 | 68.11     | 50.34 |
| Qwen1.5-7B           | vanilla | 46.67/77.53          | 56.39     | 53.98      | 54.00 | 65.98     | 44.88 |
| -                    | Ruler   | 47.27/76.68          | 56.46     | 50.18      | 54.59 | 65.19     | 47.01 |
