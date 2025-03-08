import subprocess
import json
import os

class MissingUpdatedPrompt(Exception):
    pass

class MissingPrediction(Exception):
    pass

class InvalidPrediction(Exception):
    pass

class UnknownModel(Exception):
    pass

def ollama_prediction(model, prompt, valid_predictions):
    command = ["ollama", "run", model, "--format=json", "--verbose"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # This ensures text mode instead of bytes
    )

    prompt += """
Output in JSON format like this:
    {
        "narrative_text": "...",
        "prediction": "..."
    }

`narrative_text` is where you describe your thinking process in evaluating the prompt.
`prediction` is either """ + (" or ".join(valid_predictions)) + """."""

    # Send the prompt and get outputs
    stdout, stderr = process.communicate(input=prompt)
    print(stdout)
    try:
        answer = json.loads(stdout)
    except json.decoder.JSONDecodeError:
        # If I hadn't already printed it, I would print it here
        sys.stderr.write("Non-JSON returned\n")
        raise InvalidPrediction
    if 'prediction' not in answer:
        sys.stderr.write(f"There was no prediction. The keys were {list(answer.keys())}\n")
        raise MissingPrediction
    if answer['prediction'] is None:
        sys.stderr.write(f"Prediction was null\n")
        raise MissingPrediction
    if answer['prediction'] not in valid_predictions:
        sys.stderr.write(f"Prediction was not a valid prediction: {answer['prediction']}\n")
        # Maybe there might be a way out of this
        rescued = False
        for valid in valid_predictions:
            if answer['prediction'].lower() == valid.lower():
                answer['prediction'] = valid
                rescued = True
                break
        if not rescued:
            sys.stderr.write("Could not match that case-insensitively to any valid prediction\n")
            raise InvalidPrediction
    if 'narrative_text' not in answer:
        sys.stderr.write("No narrative text\n")
        answer['narrative_text'] = ''
    #print(f"stdout = {json.dumps(answer,indent=4)}")
    info_start = stderr.index("total duration:")
    stderr = stderr[info_start:]
    print(f"stderr = {stderr}")
    return answer, stderr


def ollama_reprompt(model, prompting_creation_prompt):
    prompting_creation_prompt +=     """
Supply your answer in JSON format like this:

{
    "reasoning": "...",
    "updated_prompt": "..."
}

    Where `reasoning` explains why you are making the change and `updated_prompt` is the prompt that you think we should run next.
"""

    command = ["ollama", "run", model, "--format=json", "--verbose"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # This ensures text mode instead of bytes
    )
    # Send the prompt and get outputs
    stdout, stderr = process.communicate(input=prompting_creation_prompt)
    answer = json.loads(stdout)
    if 'updated_prompt' not in answer:
        sys.exit(f"No updated prompt supplied: {answer}")
    info_start = stderr.index("total duration:")
    stderr = stderr[info_start:]
    return answer, stderr


def claude_prediction(model, prompt, valid_predictions):
    import anthropic
    # model = "claude-3-5-haiku-20241022",
    tool_schema =  {
        "type": "function",
        "function": {
            "name": "store_prediction",
            "description": "Store the prediction",
            "input_schema": {
                "type": "object",
                "properties": {
                    "narrative_text": {
                        "type": "string",
                        "description": "Your thinking process in evaluating the prompt."
                    },
                    "prediction": {
                        "type": "string",
                        "description": "Either " + (" or ".join(valid_predictions))
                    }
                },
                "required": ["narrative_text", "prediction"]
            }
        }
    }
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    # Initialize the Anthropic client
    api_key = open(os.path.expanduser("~/.anthropic.key")).read().strip()
    client = anthropic.Client(api_key=api_key)

    response = client.messages.create(
            model=model,
        system="Use the store_prediction function to provide your analysis. You must include both a narrative_text explaining your thinking and a prediction.",
        max_tokens=8192,
            temperature=0,
            messages=messages,
            tools=[tool_schema['function']],
            tool_choice = {'name': 'store_prediction', 'type': 'tool', 'disable_parallel_tool_use': True }
        )
    #print(response.content[0].input)
    print(response)
    usage = response.usage
    if 'haiku' in model:
        cost = (0.8 * usage.input_tokens + 4 * usage.output_tokens) / 1000000
    elif 'sonnet' in model:
        cost = (3 * usage.input_tokens + 15 * usage.output_tokens) / 1000000
    else:
        cost = None
    usage_obj = {'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens, 'cost': cost}
    tool_call = response.content[0].input
    if 'narrative_text' not in tool_call:
        tool_call['narrative_text'] = ''
    return tool_call, json.dumps(usage_obj)


def claude_reprompt(model, prompting_creation_prompt):
    import anthropic
    tool_schema = {
        "type": "function",
        "function": {
            "name": "store_replacement_prompt",
            "description": "Store the new prompt",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why you are making the change."
                    },
                    "updated_prompt": {
                        "type": "string",
                        "description": "The prompt that you think we should run next",
                    }
                },
                "required": ["reasoning", "updated_prompt"]
            }
        }
    }
    messages = [
        {
            "role": "user",
            "content": prompting_creation_prompt
        }
    ]
    # Initialize the Anthropic client
    api_key = open(os.path.expanduser("~/.anthropic.key")).read().strip()
    client = anthropic.Client(api_key=api_key)

    response = client.messages.create(
        model=model,
        system="Use the store_prediction function to provide your analysis. You must include both a narrative_text explaining your thinking and a prediction.",
        max_tokens=8192,
        temperature=0,
        messages=messages,
        tools=[tool_schema['function']],
        tool_choice = {'name': 'store_replacement_prompt', 'type': 'tool', 'disable_parallel_tool_use': True }
    )
    #print(response.content[0].input)
    print(response)
    usage = response.usage
    if 'haiku' in model:
        cost = (0.8 * usage.input_tokens + 4 * usage.output_tokens) / 1000000
    elif 'sonnet' in model:
        cost = (3 * usage.input_tokens + 15 * usage.output_tokens) / 1000000
    else:
        cost = None
    usage_obj = {'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens, 'cost': cost}
    tool_call = response.content[0].input
    if 'updated_prompt' not in tool_call:
        raise MissingUpdatedPrompt
    return tool_call, json.dumps(usage_obj)


def openai_prediction(model, prompt, valid_predictions):
    from openai import OpenAI

    client = OpenAI(api_key=open(os.path.expanduser("~/.openai.key")).read().strip())
    import json
    import sys

    # Define the function schema for making a prediction.
    prediction_function = {
        "type": "function",
        "function": {
            "name": "store_prediction",
            "strict": True,
            "description": "Store the prediction along with the narrative of your thinking process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "narrative_text": {
                        "type": "string",
                        "description": "Your thinking process in evaluating the prompt."
                    },
                    "prediction": {
                        "type": "string",
                        "description": "Either " + " or ".join(valid_predictions)
                    }
                },
                "required": ["narrative_text", "prediction"],
                "additionalProperties": False
            }
        }
    }

    # Build the message list.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful prediction assistant. Use the store_prediction function to provide your answer."
        },
        {"role": "user", "content": prompt}
    ]

    # Call the OpenAI API with function calling.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[prediction_function],
        tool_choice={'type': 'function', 'function': {"name": "store_prediction"}},
        temperature=0
    )


    # Debug print (can remove if not needed)
    #print(response)
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    message = response.choices[0].message

    answer = json.loads(message.tool_calls[0].function.arguments)

    if 'prediction' not in answer:
        sys.stderr.write(f"There was no prediction. The keys were {list(answer.keys())}\n")
        raise MissingPrediction
    if answer.get("prediction") not in valid_predictions:
        sys.stderr.write(f"Prediction was not a valid prediction: {answer['prediction']}\n")
        raise MissingPrediction
    if "narrative_text" not in answer:
        sys.stderr.write("Missing narrative text\n")
        answer["narrative_text"] = ""

    if 'gpt-4o-mini' in model:
        cost = (0.15 * prompt_tokens + 0.6 * completion_tokens) / 1000000
    elif 'gpt-4o' in model:
        cost = (2.5 * prompt_tokens + 10 * completion_tokens) / 1000000
    elif 'o1' in model:
        cost = (15 * prompt_tokens + 60 * completion_tokens) / 1000000
    else:
        cost = None
    usage_obj = {'input_tokens': usage.prompt_tokens, 'output_tokens': usage.completion_tokens, 'cost': cost}
    return answer, json.dumps(usage_obj)


def openai_reprompt(model, prompting_creation_prompt):
    from openai import OpenAI

    client = OpenAI(api_key=open(os.path.expanduser("~/.openai.key")).read().strip())
    import json
    import sys

    # Define the function schema for creating a new prompt.
    reprompt_tool = {
        "type": "function",
        "function": {
            "name": "store_replacement_prompt",
            "description": "Store the updated prompt along with reasoning for the change.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why you are making the change."
                },
                "updated_prompt": {
                    "type": "string",
                    "description": "The prompt that you think we should run next."
                }
                },
                "required": ["reasoning", "updated_prompt"],
                "additionalProperties": False
            }
        }
    }

    messages = [
        {
            "role": "system",
            "content": "You are an assistant tasked with updating the prompt. Use the store_replacement_prompt function to provide your answer."
        },
        {"role": "user", "content": prompting_creation_prompt}
    ]

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              tools=[reprompt_tool],
                                              tool_choice={"type": "function", "function": {"name": "store_replacement_prompt"}})
    print(response)
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    message = response.choices[0].message
    answer = json.loads(message.tool_calls[0].function.arguments)

    if "updated_prompt" not in answer:
        sys.exit(f"No updated prompt supplied: {answer}")

    if 'gpt-4o-mini' in model:
        cost = (0.15 * prompt_tokens + 0.6 * completion_tokens) / 1000000
    elif 'gpt-4o' in model:
        cost = (2.5 * prompt_tokens + 10 * completion_tokens) / 1000000
    elif 'o1' in model:
        cost = (15 * prompt_tokens + 60 * completion_tokens) / 1000000        
    else:
        cost = None
    usage_obj = {'input_tokens': usage.prompt_tokens, 'output_tokens': usage.completion_tokens, 'cost': cost}
    return answer, json.dumps(usage_obj)



def dispatch_prediction_prompt(model, prompt, valid_predictions):
    if model in ['phi4:latest', 'llama3.3:latest', 'falcon3:1b', 'falcon3:10b', 'gemma2:27b', 'gemma2:2b', 'phi4-mini']:
        return ollama_prediction(model, prompt, valid_predictions)
    if model in ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]:
        return claude_prediction(model, prompt, valid_predictions)
    if model in ["gpt-4o", "gpt-4o-mini", 'o1']:
        return openai_prediction(model, prompt, valid_predictions)
    raise UnknownModel


def dispatch_reprompt_prompt(model, prompting_creation_prompt):
    if model in ['phi4:latest', 'llama3.3:latest', 'falcon3:1b', 'falcon3:10b', 'gemma2:27b', 'gemma2:2b', 'phi4-mini']:
        return ollama_reprompt(model, prompting_creation_prompt)
    if model in ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]:
        return claude_reprompt(model, prompting_creation_prompt)
    if model in ["gpt-4o", "gpt-4o-mini", 'o1']:
        return openai_reprompt(model, prompting_creation_prompt)
    raise UnknownModel
