# -*- coding: utf-8 -*-

import json
import requests


def do_post(api_url, query_body):
    # headers = {'content-type': 'application/json'}
    response = requests.post(api_url, json=query_body)
    json_res = json.loads(response.text)
    print(json_res)


def do_get(api_url):
    response = requests.get(api_url)
    json_res = json.loads(response.text)
    print(json_res)


# do_get_url = "http://127.0.0.1:8001/operations/c5a09848-c488-4fe3-b818-904b47063743"
# do_get(do_get_url)

query_api_url = "http://127.0.0.1:8000/query"
query_body = {
    "siteId": "10000",
    "query": "User:How to make field wire connections for the Rain Bird IVMSD?, please give me the specific answer, video link and answer start time",
    "context": {
        "subject": "",
        "history": []
    }
}
do_post(query_api_url, query_body)
