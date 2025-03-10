#!/usr/bin/env python3

import sqlite3
import os
import json

if __name__ == '__main__':
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=default_database,
                        required = default_database is None, help="Path to the SQLite database file")
    # In the future, I'd like to be able to find the round that with --patience=x --validation
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument("--encoding-instructions", required=True, help="The conversions/* text that you passed to obfuscation_plan_generator.py")
    parser.add_argument("--model", default="gpt-4o", help="What AI model will do the untranslating")
    parser.add_argument("--output-file", help="Where to put the output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    encoding_instructions = open(args.encoding_instructions).read()
    conn = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (args.round_id,))
    row = cur.fetchone()

    if row is None:
        sys.exit(f"Round ID {round_id} not found")

    encoded_prompt = row[0]
    if args.verbose:
        print("Prompt that was used")
        print("--------------------")
        print()
        print(encoded_prompt)
    from openai import OpenAI
    client = OpenAI(api_key=open(os.path.expanduser("~/.openai.key")).read().strip())
    prompt = f"""To run my experiment, I encoded my data using these encoding instructions:\n\n```\n{encoding_instructions}\n```\n\nThe prompt that was used in the experiment was this:\n\n```\n{encoded_prompt}\n```\n\nWhat would the prompt have been if had been using the original data? i.e. do the inverse of the operations from the encoding instructions to the text of the experiment's prompt. Note that the target of the rule (what is being predicted) has also been transformed."""
    messages = [{
        "role": "user",
        "content": prompt
    }]


    decoder_function = {
        "type": "function",
        "function": {
            "name": "store_decoding",
            "strict": True,
            "description": "Store the decoded prompt along with the narrative of your thinking process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "narrative_text": {
                        "type": "string",
                        "description": "Your thinking process in evaluating the prompt in the light of the code that was used to transform the data."
                    },
                    "decoded_prompt": {
                        "type": "string",
                        "description": "The prompt that would have been equivalent had it been used on the original data"
                    }
                },
                "required": ["narrative_text", "decoded_prompt"],
                "additionalProperties": False
            }
        }
    }
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=[decoder_function],
        tool_choice={'type': 'function', 'function': {"name": "store_decoding"}},
        temperature=0
    )
    answer = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    if args.verbose:
        print("")
        print("-" * 70)
    if args.verbose or not args.output_file:
        print(json.dumps(answer, indent=2))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(json.dumps(answer, indent=2))
