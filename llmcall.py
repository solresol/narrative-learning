import subprocess
import json
import os

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
    answer = json.loads(stdout)
    if answer['prediction'] not in ['Success', 'Failure']:
        # We didn't do anything. Leave it for now, and hopefully we'll come
        # back in another round
        sys.stderr.write("Invalid prediction\n")
        return
    if 'narrative_text' not in answer:
        sys.stderr.write("No narrative text\n")
        answer['narrative_text'] = ''
    #print(f"stdout = {json.dumps(answer,indent=4)}")
    info_start = stderr.index("total duration:")
    stderr = stderr[info_start:]
    print(f"stderr = {stderr}")
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
    tool_call = response.content[0].input
    if 'narrative_text' not in tool_call:
        raise KeyError
    return tool_call, None


def dispatch_prediction_prompt(model, prompt, valid_predictions):
    if model in ['phi4:latest']:
        return ollama_prediction(model, prompt, valid_predictions)
    if model in ["claude-3-5-haiku-20241022"]:
        return claude_prediction(model, prompt, valid_predictions)
    raise KeyError(model)




