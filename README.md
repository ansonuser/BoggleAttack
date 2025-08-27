
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
### Project Structure
- main.py: Main entry point. Runs attack simulations in "tree" or "liner" mode.
- playing.py: Core logic for simulating attacks and managing rounds.
- schema.py: Data models for attack, refusal, and scoring.
- template.py: Prompt templates for rephrasing and attacking.
- utils.py: Utility functions and tree node management.
- model_list.py: Supported model configurations.
- sentence.py: List of target prompts for attack.
- openai_model.py: Model interface and API handling.

### How to Run

1. Install dependencies (see requirements.txt if available).
2. Run the main attack script:

```python
python boggle/attacks/main.py --mode tree
```
- Use --mode liner for linear attack mode.

3. Results are saved as Excel files (attack_result_tree_bulk.xlsx or attack_result_liner.xlsx).

### Example Target Prompts
See sentence.py for a list of prompts used to test model responses.

### Disclaimer
This project is for research and educational purposes only. Do not use it to generate or disseminate harmful content.