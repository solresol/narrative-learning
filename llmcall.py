import subprocess
import json
import os
import sys

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
    try:
        info_start = stderr.index("total duration:")
    except ValueError:
        sys.stderr.write("stderr was in an unusual format\n" + stderr + "\n")
        info_start = 0
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
    try:
        answer = json.loads(stdout)
    except json.decoder.JSONDecodeError:
        sys.stderr.write(f"Did not get a JSON response:\n{stdout}\n")
        sys.stderr.write(f"STDERR was:\n{stderr}\n")
        sys.exit(1)
    if 'updated_prompt' not in answer:
        sys.exit(f"No updated prompt supplied: {answer}")
    try:
        info_start = stderr.index("total duration:")
    except ValueError:
        sys.stderr.write("stderr was in an unusual format:\n" + stderr + "\n")
        info_start = 0
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
        #temperature=0,
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
        #temperature=0,
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
        tool_choice={'type': 'function', 'function': {"name": "store_prediction"}}
        #temperature=0
    )


    # Debug print (can remove if not needed)
    #print(response)
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    message = response.choices[0].message

    try:
       answer = json.loads(message.tool_calls[0].function.arguments)
    except json.decoder.JSONDecodeError:
       sys.stderr.write(f"Received something that wasn't valid JSON: {message.tool_calls[0].function.arguments}\n")
       raise MissingPrediction

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



def gemini_prediction(model, prompt, valid_predictions):
    from google import genai
    from google.genai import types
    
    # Initialize the Gemini client
    api_key = open(os.path.expanduser("~/.gemini.key")).read().strip()
    client = genai.Client(api_key=api_key)
    
    def store_prediction(narrative_text: str, prediction: str) -> dict:
        """Store the prediction
        
        Args:
            narrative_text: Your thinking process in evaluating the prompt.
            prediction: Either """ + " or ".join(valid_predictions) + """
        """
        return {"narrative_text": narrative_text, "prediction": prediction}
    
    prompt += """
Use the tool provided to submit your answer. You must include both a narrative_text explaining your thinking and a prediction.
"""
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[store_prediction]
        ),
    )
    
    print(response)
    
    # Extract function call results
    function_call = response.candidates[0].content.parts[0].function_call
    tool_call = {
        "narrative_text": function_call.args["narrative_text"],
        "prediction": function_call.args["prediction"]
    }
    
    if 'prediction' not in tool_call:
        sys.stderr.write(f"There was no prediction. The keys were {list(tool_call.keys())}\n")
        raise MissingPrediction
        
    if tool_call.get("prediction") not in valid_predictions:
        sys.stderr.write(f"Prediction was not a valid prediction: {tool_call['prediction']}\n")
        # Maybe there might be a way out of this
        rescued = False
        for valid in valid_predictions:
            if tool_call['prediction'].lower() == valid.lower():
                tool_call['prediction'] = valid
                rescued = True
                break
        if not rescued:
            sys.stderr.write("Could not match that case-insensitively to any valid prediction\n")
            raise InvalidPrediction
            
    if "narrative_text" not in tool_call:
        sys.stderr.write("Missing narrative text\n")
        tool_call["narrative_text"] = ""
    
    # Calculate usage and cost - will need to be updated when proper usage stats are available
    usage_obj = {
        'input_tokens': 0,  # Placeholder
        'output_tokens': 0, # Placeholder
        'cost': None        # Placeholder
    }
    
    return tool_call, json.dumps(usage_obj)


def dispatch_prediction_prompt(model, prompt, valid_predictions):
    if model in ['phi4:latest', 'llama3.3:latest', 'falcon3:1b', 'falcon3:10b', 'gemma2:27b', 'gemma2:2b', 'phi4-mini', 'deepseek-r1:70b', 'qwq:32b', 'gemma3:27b', 'cogito:70b']:
        return ollama_prediction(model, prompt, valid_predictions)
    if model in ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"]:
        return claude_prediction(model, prompt, valid_predictions)
    if model in ["gpt-4o", "gpt-4o-mini", 'o1', 'o3', 'gpt-4.1', 'gpt-4.5-preview', 'gpt-3.5-turbo']:
        return openai_prediction(model, prompt, valid_predictions)
    if model in ["gemini-2.0-flash", "gemini-2.0-pro", "gemma-3-27b-it", "gemini-2.0-pro-exp", "gemini-2.5-pro-exp-03-25"]:
        return gemini_prediction(model, prompt, valid_predictions)
    raise UnknownModel


def gemini_reprompt(model, prompting_creation_prompt):
    from google import genai
    from google.genai import types
    
    # Initialize the Gemini client
    api_key = open(os.path.expanduser("~/.gemini.key")).read().strip()
    client = genai.Client(api_key=api_key)
    
    def store_replacement_prompt(reasoning: str, updated_prompt: str) -> dict:
        """Store the updated prompt along with reasoning for the change.
        
        Args:
            reasoning: Why you are making the change.
            updated_prompt: The prompt that you think we should run next.
        """
        return {"reasoning": reasoning, "updated_prompt": updated_prompt}
    
    prompting_creation_prompt += """
Supply your answer using the tool provided. You must include both reasoning explaining why you are making the change and an updated_prompt containing the prompt that should be run next.
"""
    
    response = client.models.generate_content(
        model=model,
        contents=prompting_creation_prompt,
        config=types.GenerateContentConfig(
            tools=[store_replacement_prompt],
            automatic_function_calling= {'disable': True},
            tool_config = {
                  'function_calling_config': {
                  'mode': 'any'
                }
             }
        ),
    )
    
    print(response)
    
    # Extract function call results
    function_call = response.candidates[0].content.parts[0].function_call
    tool_call = {
        "reasoning": function_call.args["reasoning"],
        "updated_prompt": function_call.args["updated_prompt"]
    }
    
    if 'updated_prompt' not in tool_call:
        raise MissingUpdatedPrompt
    
    # Calculate usage and cost - will need to be updated when proper usage stats are available
    usage_obj = {
        'input_tokens': 0,  # Placeholder
        'output_tokens': 0, # Placeholder
        'cost': None        # Placeholder
    }
    
    return tool_call, json.dumps(usage_obj)


def dispatch_reprompt_prompt(model, prompting_creation_prompt):
    if model in ['phi4:latest', 'llama3.3:latest', 'falcon3:1b', 'falcon3:10b', 'gemma2:27b', 'gemma2:2b', 'phi4-mini', 'deepseek-r1:70b', 'qwq:32b', 'gemma3:27b', 'cogito:70b']:
        return ollama_reprompt(model, prompting_creation_prompt)
    if model in ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"]:
        return claude_reprompt(model, prompting_creation_prompt)
    if model in ["gpt-4o", "gpt-4o-mini", 'o1',  'o3', 'gpt-4.1', 'gpt-4.5-preview', 'gpt-3.5-turbo']:
        return openai_reprompt(model, prompting_creation_prompt)
    if model in ["gemini-2.0-flash", "gemini-2.0-pro", "gemma-3-27b-it", "gemini-2.0-pro-exp", "gemini-2.5-pro-exp-03-25"]:
        return gemini_reprompt(model, prompting_creation_prompt)
    sys.stderr.write(f"{model}\n")
    raise UnknownModel



def sanity_check_prompt(prompt, sample, valid_answers):
    # For speed, skipping the check
    return True
    from openai import OpenAI

    client = OpenAI(api_key=open(os.path.expanduser("~/.openai.key")).read().strip())
    import json
    import sys

    # Define the function schema for creating a new prompt.
    quality_check_doc = {
        "type": "function",
        "function": {
            "name": "store_quality_check",
            "description": "Store the details of the prompt quality check",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "is_valid": {
                        "type": "boolean",
                        "description": "True if the prompt supplied is coherent and able to be used for inference on the data sample."
                    }
                },
                "required": ["is_valid"],
                "additionalProperties": False
            }
        }
    }

    messages = [
        {
            "role": "system",
            "content": "You are an assistant doing a quality check on a prompt that will be used later. The question is whether the prompt is a coherent set of rules that can be used to predict where the data point given is '" + ("' or '".join(valid_answers)) + "'."
        },
        {"role": "user", "content": f"""
We are about to launch a large and expensive run where we use the following prompt over
a huge number of texts. Due to the structure of the task, no human being can review this
prompt before it is used. We want to make sure that the rules supplied are reproducible and
relevant. They need to be clear, and uniquely define how to choose between the valid
outputs:

  - {'\n - '.join(valid_answers)}

Here is the prompt:\
```
{prompt}
```

Now, here is some sample data that is very similar to the kinds of texts that the
prompt will be used on

```
{sample}
```

Give your opinion: if you had just received that prompt and that sample, would you be
confidently able to give a clear answer by following those rules? Or do we need to
get the prompt re-written before we deploy it into production?
"""}
    ]

    response = client.chat.completions.create(model='gpt-4o-mini',
                                              messages=messages,
                                              tools=[quality_check_doc],
                                              tool_choice={"type": "function", "function": {"name": "store_quality_check"}})
    #print(response)
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    message = response.choices[0].message
    answer = json.loads(message.tool_calls[0].function.arguments)

    if "is_valid" not in answer:
        sys.exit(f"Couldn't even evaluate whether the prompt was any good.")

    return answer['is_valid']
