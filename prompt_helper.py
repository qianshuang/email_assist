# -*- coding: utf-8 -*-

def gen_chat_his_summary_prompt(chat_his):
    prompt_str = """GOAL:
Create a concise running summary based on the DIALOGUE below, focusing on key and potentially important information to remember.
Keep the summary concise, within 400 words.

DIALOGUE:
{}

DIALOGUE Summary:
""".format(chat_his)

    return [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": prompt_str}]


def gen_standalone_question_prompt(transed_dialogue, question):
    prompt_str = """GOAL:
Given the DIALOGUE and Follow Up Question below, rephrase the Follow Up Question to be a Standalone Question, focusing on key and potentially important information. Do not lose any user demands in the Follow Up Question.

DIALOGUE:
{}

Follow Up Question:
{}

Standalone Question:
""".format(transed_dialogue, question)

    return [{"role": "system", "content": "You are a trustworthy AI assistant."},
            {"role": "user", "content": prompt_str}]


def gen_retrieve_prompt_with_standalone(remain_ctx, standalone_question, email_history):
    prompt_str = """GOAL:
With reference to the CONTEXT (each EVENT is separated by '------------') below, generate the most appropriate answer to the Standalone Question based on the DIALOGUE below.
Let's work this out in a step by step way to be sure we have the right answer.

CONTEXT:
{}

Standalone Question:
{}

DIALOGUE:
{}
AI:""".format(remain_ctx, standalone_question, email_history)

    sys_content = """You are a trustworthy AI assistant.
Remember that you should figure out the answer only from the DIALOGUE and CONTEXT (use the original text in the DIALOGUE and CONTEXT as much as possible).
Please try your best to provide comprehensive responses to all the user's questions, without omitting any. If you can't find the answer to a particular question, simply reply 'I don't know' instead of attempting to fabricate an answer.
Please reply in the tone of the characters in the DIALOGUE."""
    return [{"role": "system", "content": sys_content},
            {"role": "user", "content": prompt_str}]
