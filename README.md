# SuperUAT

Implementation of our paper "**Universal and Transferable Adversarial Attacks on Prompt-based Language Models**" 


## Overview

In prompt-based learning, pre-trained language models (PLMs) are fine-tuned with a small set of prompts, leading to the creation of Prompt-based Fine-tuned Models (PFMs) for specific downstream tasks.

- **SuperUAT** : A transferable and universal adversarial attack algorithm on PFMs, with an alternative optimization method, which randomly selects and optimizes tokens across the entire trigger, providing a more effective search process and achieving superior attack success rates.

## Installation

Please install `pytorch>=1.8.0` and correctly configure the GPU accelerator. (**GPU is required.**)

Install all requirements by

```
pip install -r requirements.txt
```

## Usage

### SuperUAT: Transferable and Universal Adversarial Attack on PFMs

`search_uat.py` implements the trigger search on RoBERTa-large model. 

**command:**

```
python3 -m search_uat.py --trigger_len 3 --trigger_pos all
```

To search for position-sensitive triggers, you can change `--trigger_pos` to 

- `prefix`: the trigger is supposed to be placed before the text.
- `suffix`: the trigger is supposed to be placed after the text.

**output:**
Results will be stored in the `triggers/` folder as a JSON file.

### Evaluation

`eval.py` can evaluate the performance of the SuperUAT attack.

**Evaluate SuperUAT**

```
python3 -m eval --shots 16 --dataset ag_news --target_label -1 \
	--repeat 5 --bert_type roberta-large --template_id 0 --load_trigger trigger/<trigger_json>
```

The arguments:

- `dataset`: The evaluation datasets. 
- `shots`: The number of samples per label.
- `target_label`: 
	- set -1 in our experiment settings
- `bert_type`: The type of PLMs. 
- `template_id`: The chosen template. Check `prompt/` folder for all prompt templates.

**Important note**: 

- for `all`-purpose triggers, you can choose template_id from \{0, 1, 2, 3\}.
- for `prefix` triggers, you can choose template_id from \{0, 2}.
- for `suffix` triggers, you can choose template_id from \{1, 3}.
- We also provide evaluation on llama2 model in `eval_llama2.py` and GPT model in `eval_GPT.py`. The implementation of the evaluation is similar to `eval.py`. Note that you need to modify the path in these files to the correct model path. Additionally, the `eval_llama2.py` have a stricter limit on your GPU memory, and `eval_GPT.py` needs your own GPT api if you need to evaluate on GPT model.

Use `python3 -m eval --help` for details.


## Datasets and Prompts

The datasets and prompts used in experiments are in `data/` and `prompt/` folders.


