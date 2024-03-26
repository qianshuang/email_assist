# -*- coding:utf-8 -*-

import json

import requests


def call_azure(prompt, api_url="https://comm100gpttest.openai.azure.com/openai/deployments/GPTChat/chat/completions?api-version=2023-03-15-preview", key="e5fda94edc774de7b4fa75a663a7cd5a"):
    query_body = {"temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    headers = {'content-type': 'application/json', 'api-key': key}
    response = requests.post(api_url, json=query_body, headers=headers, timeout=60)
    json_res = json.loads(response.text)
    return json_res["choices"][0]["message"]["content"]


def gen_standalone_question_prompt(transed_dialogue, question):
    prompt_str = """GOAL:
Given the DIALOGUE and Follow Up Question below, rephrase the Follow Up Question to be a Standalone Question, focusing on key and potentially important information.

DIALOGUE:
{}

Follow Up Question:
{}

Standalone Question:
""".format(transed_dialogue, question)

    return [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": prompt_str}]


p_ = """GOAL:
Given the DIALOGUE and Follow Up Question below, rephrase the Follow Up Question to be a Standalone Question, focusing on key and potentially important information.

DIALOGUE:


Follow Up Question:
Who are you? What is your name? What is your age? Is there any books with you?

Standalone Question:
"""
print(call_azure(p_))
