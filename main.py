# -*- coding: utf-8 -*-

import datetime
import uuid
import redis_lock

from prompt_helper import *
from data_process_helper import *
from emb_helper import *

from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()


class Context(BaseModel):
    subject: str
    history: List[str]


class QueryBody(BaseModel):
    siteId: str
    query: str
    context: Context


class CreateBody(BaseModel):
    siteId: int  # 每个site代表一个客户
    emailHistoryDataUrl: Optional[str]
    chatHistoryDataUrl: Optional[str]
    botDataUrl: Optional[str]
    kbDataUrl: Optional[str]
    websiteUrl: Optional[str]
    customDataUrl: Optional[list]


def build_site(cb, bot_path, request_id, bot_lock):
    r.set(request_id, "BUILDING")
    # 1. 下载数据（全量覆盖）
    # 2. 索引构建
    try:
        if cb.emailHistoryDataUrl is not None and cb.emailHistoryDataUrl.strip() != "":
            save_path = download_data(cb.emailHistoryDataUrl.strip(), os.path.join(bot_path, "email_history_data.jsonl"))
            process_eml_his_data(save_path)
        if cb.chatHistoryDataUrl is not None and cb.chatHistoryDataUrl.strip() != "":
            save_path = download_data(cb.chatHistoryDataUrl.strip(), os.path.join(bot_path, "chat_history_data.jsonl"))
            process_chat_his_data(save_path)
        if cb.botDataUrl is not None and cb.botDataUrl.strip() != "":
            save_path = download_data(cb.botDataUrl.strip(), os.path.join(bot_path, "bot_data.jsonl"))
            process_bot_data(save_path)
        if cb.kbDataUrl is not None and cb.kbDataUrl.strip() != "":
            save_path = download_data(cb.kbDataUrl.strip(), os.path.join(bot_path, "kb_data.jsonl"))
            process_kb_data(save_path)
        if cb.customDataUrl is not None and len(cb.customDataUrl) > 0:
            file_content_dic = download_extract_data(cb.customDataUrl, bot_path)
            process_custom_data(file_content_dic, bot_path)
        if cb.websiteUrl is not None and cb.websiteUrl.strip() != "":
            file_content_dic = extract_html("data/html_pages")
            process_spider_data(file_content_dic, bot_path)

        logger.info("build_site: {} finished.".format(bot_path))
        r.set(request_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        bot_lock.release()
    except:
        logger.exception("build_site: {} failed.".format(bot_path))
        r.set(request_id, "ERROR")
        bot_lock.release()


@app.post("/upd_cre_ate")
def upd_cre_ate(cb: CreateBody, background_task: BackgroundTasks):
    # 创建workspace
    bot_name = "site_" + str(cb.siteId)

    bot_lock = redis_lock.Lock(r, "lock_" + bot_name)
    acquired_bot_lock = bot_lock.acquire(blocking=False)
    if not acquired_bot_lock:
        return {'code': -3, 'msg': 'someone is refreshing this site, please wait.'}

    bot_path = os.path.join(BOT_SRC_DIR, bot_name)
    if not os.path.exists(bot_path):
        os.makedirs(bot_path)

    request_id = str(uuid.uuid4())
    r.set(request_id, "PENDING")
    background_task.add_task(build_site, cb, bot_path, request_id, bot_lock)
    return {'code': 0, 'msg': 'success', 'data': {"status": r.get(request_id).decode(), "operation_id": request_id}}


@app.get("/operations/{operation_id}")
def get_status(operation_id: str):
    if not r.exists(operation_id):
        return {'code': -4, 'msg': "operation_id: {} doesn't exist.".format(operation_id)}
    status = r.get(operation_id).decode()
    if "-" in status:
        return {'code': 0, 'msg': 'success', 'data': {"operation_id": operation_id, "status": "FINISHED", "finish_time": status}}
    else:
        return {'code': 0, 'msg': 'success', 'data': {"operation_id": operation_id, "status": status}}


@app.post("/query")
def query(qb: QueryBody):
    print("{} is processing...".format(os.getpid()))
    bot_name = "site_" + str(qb.siteId)
    bot_path = os.path.join(BOT_SRC_DIR, bot_name)
    if not os.path.exists(bot_path):
        return {'code': -2, 'msg': "site {} agent doesn't exist...".format(qb.siteId)}

    try:
        # 记忆填充
        origin_history = qb.context.history.copy()
        qb.context.history.append(qb.query)

        json_obj = qb.context.model_dump()
        if "Subject:" in json_obj["subject"]:
            content4emb, email_history = extract_email_history(json_obj)
        else:
            content4emb, email_history = extract_query(json_obj)

        email_history = filter_empty_mess(email_history)
        truncated = email_history[:-3]
        email_history = email_history[-3:]  # 取最近3轮

        # runtime memory summary
        if len(truncated) > 0:
            chat_his_summary_prompt = gen_chat_his_summary_prompt("\n".join(truncated))
            chat_his_summary = get_gpt_downgrade_res(chat_his_summary_prompt)
            email_history.insert(0, "Dialogue History Summary:" + chat_his_summary)
        email_history = "\n".join(email_history).strip()

        standalone_question_prompt = gen_standalone_question_prompt("\n".join(origin_history), qb.query)
        logger.info(standalone_question_prompt[1]["content"])

        standalone_question = get_gpt_downgrade_res(standalone_question_prompt)
        # TODO
        # 1. 全文检索
        # 2. 分层检索
        content4emb = "\n".join(content4emb).strip()

        embs = get_emb([standalone_question])
        embs = np.array(embs).astype('float32')

        # Context生成
        all_context = []
        all_src = []

        # 1. EMAIL_HISTORY_EMB
        eml_his_emb_path = os.path.join(bot_path, EMAIL_HISTORY_EMB)
        if os.path.exists(eml_his_emb_path + ".faiss"):
            eml_context, sims = gen_emb_context(embs, eml_his_emb_path, eml_emb_sim_threshold, eml_emb_sim_cand_num)
            for i, seg in enumerate(eml_context):
                all_context.append(seg)
                all_src.append({"file": "email history", "segment": seg, "score": sims[i]})
        # 2. CHAT_HISTORY_EMB
        chat_his_emb_path = os.path.join(bot_path, CHAT_HISTORY_EMB)
        if os.path.exists(chat_his_emb_path + ".faiss"):
            chat_context, sims = gen_emb_context(embs, chat_his_emb_path, chat_emb_sim_threshold, chat_emb_sim_cand_num)
            for i, seg in enumerate(chat_context):
                all_context.append(seg)
                all_src.append({"file": "chat history", "segment": seg, "score": sims[i]})
        # 3. BOT_EMB
        bot_emb_path = os.path.join(bot_path, BOT_EMB)
        if os.path.exists(bot_emb_path + ".faiss"):
            bot_context, sims = gen_emb_context(embs, bot_emb_path, bot_emb_sim_threshold, bot_emb_sim_cand_num)
            for i, seg in enumerate(bot_context):
                all_context.append(seg)
                all_src.append({"file": "bot", "segment": seg, "score": sims[i]})
        # 4. KB_EMB
        kb_emb_path = os.path.join(bot_path, KB_EMB)
        if os.path.exists(kb_emb_path + ".faiss"):
            kb_context, sims = gen_emb_context(embs, kb_emb_path, kb_emb_sim_threshold, kb_emb_sim_cand_num)
            for i, seg in enumerate(kb_context):
                all_context.append(seg)
                all_src.append({"file": "kb", "segment": seg, "score": sims[i]})
        # 5. CUSTOM_EMB
        custom_emb_path = os.path.join(bot_path, CUSTOM_EMB)
        if os.path.exists(custom_emb_path + ".faiss"):
            custom_context, sims, srcs = gen_emb_context(embs, custom_emb_path, custom_emb_sim_threshold, custom_emb_sim_cand_num, "content4emb", True)
            for i, seg in enumerate(custom_context):
                all_context.append(seg)
                all_src.append({"file": srcs[i], "segment": seg, "score": sims[i]})
        # 6. SPIDER_EMB
        spider_emb_path = os.path.join(bot_path, SPIDER_EMB)
        if os.path.exists(spider_emb_path + ".faiss"):
            spider_context, sims, srcs = gen_emb_context(embs, spider_emb_path, spider_emb_sim_threshold, spider_emb_sim_cand_num, "content4emb", True)
            for i, seg in enumerate(spider_context):
                all_context.append(seg)
                all_src.append({"file": srcs[i], "segment": seg, "score": sims[i]})

        if len(all_context) == 0:
            return {'code': 0, 'msg': 'success', 'data': "Sorry, I don't know the answer."}

        all_context.reverse()
        remain_ctx, _, remain_idx = truncate_massages(all_context, 30000)
        remain_ctx = "\n------------\n".join(remain_ctx)

        # 答案生成
        prompt = gen_retrieve_prompt_with_standalone(remain_ctx, standalone_question, email_history)
        chat_res = get_gpt_downgrade_res(prompt)

        return {
            'code': 0,
            'msg': 'success',
            'data': chat_res
        }
    except:
        logger.exception("call query failed......")
        return {'code': -1, 'msg': "call query failed......"}
