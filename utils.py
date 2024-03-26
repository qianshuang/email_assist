# -*- coding: utf-8 -*-

import json
import logging

import redis
import tiktoken
from boltons.iterutils import remap
from concurrent_log import ConcurrentTimedRotatingFileHandler
import re

# 日志配置
logger = logging.getLogger("email_assist_logger")
logger.setLevel(logging.INFO)
handler = ConcurrentTimedRotatingFileHandler(
    filename="log/email_assist", when="MIDNIGHT", interval=1, backupCount=3, encoding="utf-8"
)
handler.suffix = "%Y-%m-%d.log"
handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

BOT_SRC_DIR = "bot_resources"
EMAIL_HISTORY_EMB = "email_history_emb"
CHAT_HISTORY_EMB = "chat_history_emb"
BOT_EMB = "bot_emb"
KB_EMB = "kb_emb"
CUSTOM_EMB = "custom_emb"
SPIDER_EMB = "spider_emb"

# redis配置
r = redis.Redis(db=1)


def count_token(prompt):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


def remove_dic_null(dic):
    doc_temp = remap(dic, visit=lambda path, key, value: value is not None and value != "" and value != [])
    return doc_temp


def parse_json(text):
    try:
        first_brace_index = text.find("{")
        last_brace_index = text.rfind("}")
        text = text[first_brace_index:last_brace_index + 1]

        state = json.loads(text, strict=False)
        state = remove_dic_null(state)
    except:
        try:
            result = re.sub(r'[,:.](?=\s*})', '', text)  # 匹配逗号后面紧跟着的大括号，去掉该逗号
            state = json.loads(result, strict=False)
            state = remove_dic_null(state)
        except:
            logger.error("can not parse json from: " + text)
            state = {}
    return state


def load_json_file(json_file_path):
    with open(json_file_path, 'r') as json_file:
        qa_dict = json.load(json_file)
    return qa_dict
