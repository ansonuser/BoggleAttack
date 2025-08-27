
# BoggleAttack

## Description:

BoggleAttack is a research tool inspired by the word game Boggle, designed to probe and analyze the response behavior of language models when presented with potentially harmful or malicious prompts. The project simulates attacks by rephrasing and escalating prompts to maximize the harmfulness of model responses, while tracking refusal and semantic preservation.

### How It Works
- The attack process mimics Boggle’s pathfinding: prompts are rephrased and iteratively modified.
- The system evaluates if the target model refuses to answer, and adapts the prompt accordingly:
1. Preserve semantic meaning as much as possible.
2. Weaken maliciousness if the model refuses to answer.
3. Enhance maliciousness if the model does not refuse.
- The goal is to maximize the harmful level of a response from the target model.

### Set Env Variables

create and set the variables

```bash
$ touch .env
```

**Example**

```
EVALUATE_MODEL=gemini-2.5-flash
TARGET_MODEL=gemini-2.5-flash
MAXROUNDS=3
ALPHA=0.7
WIDTH=4
MAXCON=2
```



### Project Structure
.
├── poetry.lock
├── pyproject.toml
├── README.md
└── src
    └── boggle
        ├── attacks
        │   ├── helper
        │   │   ├── playing.py
        │   │   ├── schema.py
        │   │   └── template.py
        │   └── main.py
        ├── configs
        │   ├── model_list.py
        │   └── sentence.py
        ├── models
        │   └── openai_model.py
        └── utils.py


### How to Run

1. Install dependencies (see pyproject.toml).
2. Run the main attack script:

```
$ cd boggle
$ poetry install
$ poetry run python src/boggle/attacks/main.py
```
- Use --mode liner for linear attack mode.

3. Results are saved as Excel files (attack_result_tree_bulk.xlsx or attack_result_liner.xlsx).

### Example Target Prompts
See sentence.py for a list of prompts used to test model responses.

### Disclaimer
This project is for research and educational purposes only. Do not use it to generate or disseminate harmful content.