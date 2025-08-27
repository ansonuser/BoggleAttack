import re
import json
from typing import Dict
from pydantic import BaseModel
from boggle.models.openai_model import GPTModel
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState
)
import logging
import openai
import os
from openai import AsyncOpenAI

retryable_exceptions = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.LengthFinishReasonError,
)


class TreeNode:
    def __init__(self, prompt:str, score:float, depth:int, conversation_history=None):
        self.children = []
        self.convesation_history = conversation_history or []
        self.depth = depth 
        self.prompt = prompt
        self.score = score 
        self.positive = False # next move is same direction as previous move or not
        self.response = None  # response from the model
        self.refused = False
        self.toxic_score = "Unknown"
        
    def __repr__(self):
        if self.response is None:
            self.response = "No response yet"
        return f"TreeNode(prompt={self.prompt}, score={self.score}, depth={self.depth}, positive={self.positive}, num_children={len(self.children)}, response={self.response}, refused={self.refused})"
 

def trim_and_load_json(
    input_string: str,
) -> Dict:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1
    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)
    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
    
    
def generate_schema(
    prompt: str,
    schema: BaseModel,
    model: GPTModel = None,
) -> BaseModel:
    """
    Generate schema using the provided model.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        The validated schema object
    """
    if model is None:
        raise ValueError("Model must be provided for schema generation.")
    
    try:
        res = model.generate(prompt, schema=schema)
        return res
    except TypeError:
        res = model.generate(prompt)
        data = trim_and_load_json(res)
        return schema(**data)
    
    
async def a_generate_schema(
    prompt: str,
    schema: BaseModel,
    model: GPTModel = None,
) -> BaseModel:
    """
    Asynchronously generate schema using the provided model.

    Args:
        prompt: The prompt to send to the model
        schema: The schema to validate the response against
        model: The model to use

    Returns:
        The validated schema object
    """
    try:
        res = await model.a_generate(prompt, schema=schema)
        return res
    except TypeError:
        res = await model.a_generate(prompt)
        print("fail to generate schema, using default model")
        print(res)
        data = trim_and_load_json(res)
        return schema(**data)
    
def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"OpenAI Error: {exception} Retrying: {retry_state.attempt_number} time(s)..."
    )
    
    
    
@retry(
    wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
    retry=retry_if_exception_type(retryable_exceptions),
    after=log_retry_error,
)
async def callback(input:str, model_name="gpt-4o")->str:
    client = AsyncOpenAI(base_url=os.getenv("OPENAI_API_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    prompt = [{"role": "user", "content": input}]
    completion = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": json.dumps(prompt)}],
        temperature=1.0,
        )

    output = completion.choices[0].message.content
    return output