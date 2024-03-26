# -*- coding: utf-8 -*-

import faiss
import pandas as pd

eml_emb_sim_threshold = 0.5
eml_emb_sim_cand_num = 8

chat_emb_sim_threshold = 0.5
chat_emb_sim_cand_num = 8

bot_emb_sim_threshold = 0.5
bot_emb_sim_cand_num = 8

kb_emb_sim_threshold = 0.5
kb_emb_sim_cand_num = 8

custom_emb_sim_threshold = 0.5
custom_emb_sim_cand_num = 8

spider_emb_sim_threshold = 0.5
spider_emb_sim_cand_num = 8


def gen_emb_context(query_emb, emb_path_prefix, sim_threshold, emb_cand_num, content_col_name="content", with_src=False):
    shop_faiss = emb_path_prefix + ".faiss"
    shop_txt = emb_path_prefix + ".txt"

    df_shop_chat_his = pd.read_csv(shop_txt, encoding="utf-8")
    shop_chat_his_index = faiss.read_index(shop_faiss)

    Diss, Idxs = shop_chat_his_index.search(query_emb, emb_cand_num)
    valid_idx = [ix for i, ix in enumerate(Idxs[0]) if Diss[0][i] >= sim_threshold]
    sims = [float(i) for i in Diss[0] if i >= sim_threshold]

    df_shop_chat_his = df_shop_chat_his.iloc[valid_idx]
    nearest_chats = df_shop_chat_his[content_col_name].values.tolist()

    if with_src:
        return nearest_chats, sims, df_shop_chat_his["src"].values.tolist()
    else:
        return nearest_chats, sims
