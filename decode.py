#!/usr/bin/env python3

import sqlite3
import common
import os

if __name__ == '__main__':
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=default_database,
                        required = default_database is None, help="Path to the SQLite database file")
    # In the future, I'd like to be able to find the round that with --patience=x --validation 
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument("--encoder-program", required=True, help="The python program that you used to obfuscate the data")
    parser.add_argument("--model", default="gpt-4o", help="What AI model will do the untranslating")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    encoder = open(args.encoder_program).read()
    encoded_prompt = common.get_round_prompt(conn, args.round_id)
    print(encoded_prompt)
    from openai import OpenAI
    client = OpenAI(api_key=open(os.path.expanduser("~/.openai.key")).read().strip())
    prompt = f"""To run my experiment, I encoded my data using this program:\n\n```\n{encoder}\n```\n\nThe prompt that was used in the experiment was this:\n\n```\n{encoded_prompt}\n```\n\nWhat would the prompt have been if had been using the original data? i.e. do the inverse of the operations from the program to the text of the experiment's prompt. Note that the target of the rule (what is being predicted) has also been transformed."""
    messages = [{
        "role": "user",
        "content": prompt
    }]
    response = client.chat.completions.create(
        model=args.model,
        messages=messages
    )
    print("-" * 70)
    print(response.choices[0].message.content)
