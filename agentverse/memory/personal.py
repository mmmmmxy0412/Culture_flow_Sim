"""
Personal Experience of Twitter Users
- constructed from the user's historical tweets
"""
from typing import List, Union

from pydantic import Field

from agentverse.message import Message, TwitterMessage
from agentverse.llms import BaseLLM
from agentverse.llms.openai import get_embedding, OpenAIChat


from . import memory_registry
from .base import BaseMemory
from tqdm import tqdm
import json
import os
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import numpy as np
import openai

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    # print("tokenized_doc", tokenized_doc)
    return tokenized_doc

def build_bm25_retriever(corpus):
    tokenized_corpus = []
    for passage in corpus:
        # print("passege"+passage)
        tokenized_corpus.append(bm25_tokenizer(passage))

    bm25 = BM25Okapi(tokenized_corpus)
    # print(f"BM25 retriever created - Document count: {len(bm25.doc_freqs)}")
    return bm25      

def bm25_retrieve_facts(bm25, passages, ques, top_k=10, thred=1.1):
    bm25_scores = bm25.get_scores(bm25_tokenizer(ques))
    # print("bm25_scores",bm25_scores)
    top_k = min(top_k, len(passages))
    top_n = np.argpartition(bm25_scores, -top_k)[-top_k:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    res = []
    idx = []
    for hit in bm25_hits:
        if hit['score']>=thred:
            res.append(passages[hit['corpus_id']].replace("\n", " "))
            idx.append(hit['corpus_id'])
    return res, idx 


@memory_registry.register("personal_history")
class PersonalMemory(BaseMemory):
    messages: List[Message] = Field(default=[])
    memory_path: str = None
    target: str = "participating in and sharing Chinese cultural events"
    top_k: str = 5
    deadline: str = None
    model: str = "gpt-3.5-turbo"
    has_summary: bool = False
    max_summary_length: int = 200
    summary: str = ""
    SUMMARIZATION_PROMPT = '''Your task is to create a concise running summary of observations in the provided text, focusing on key and potentially important information to remember.

Please avoid repeating the observations and pay attention to the person's overall leanings. Keep the summary concise in one sentence.

Observations:
"""
{new_events}
"""
'''
    RETRIVEAL_QUERY='''What is your opinion on {target} or other culture event?'''

    # def __init__(self, memory_path, target, top_k, deadline, llm):
    #     super().__init__()
    #     self.memory_path = memory_path
    #     self.target = target
    #     self.top_k = top_k
    #     self.deadline = deadline
    #     self.model = llm
    #     # load the historical tweets of the user
    #     if self.memory_path is not None and os.path.exists(self.memory_path):
    #         print('load ',self.memory_path)
    #         df = open(self.memory_path,'r',errors='ignore').readlines()
    #         content_set = set()
    #         for d in df:
    #             try:
    #                 d = json.loads(d)
    #             except json.JSONDecodeError:
    #                 # 如果这一行 JSON 有问题（比如引号没转义），就跳过
    #                 continue
    #
    #             # 处理新的推文结构
    #             if "title" in d and "content" in d:
    #                 # 组合title和content作为完整内容
    #                 title = d.get("title", "").strip()
    #                 content = d.get("content", "").strip()
    #                 title=title.replace("\n", " ")
    #                 content=content.replace("\n", " ")
    #                 if title and content:
    #                     full_content = f"{title}\n{content}"
    #                 elif title:
    #                     full_content = title
    #                 elif content:
    #                     full_content = content
    #                 else:
    #                     continue  # 如果title和content都为空，跳过
    #             elif "rawContent" in d:
    #                 # 向后兼容旧格式
    #                 full_content = d.get("rawContent", "").strip()
    #             else:
    #                 # 如果没有找到内容字段，跳过
    #                 continue
    #
    #             # 清理内容格式
    #             full_content = re.sub(r"\n+", "\n", full_content)
    #
    #
    #             # 去重和长度检查
    #             if full_content in content_set:
    #                 continue
    #             content_set.add(full_content)
    #
    #
    #
    #             post_time = d["date"][:19]
    #             if post_time>self.deadline:continue
    #             sender = d["user"]["username"]
    #             if sender !=self.memory_path.split('/')[-1][:-4]:continue
    #             message = TwitterMessage(content=full_content, post_time=post_time, sender=sender)
    #             # print("messages" + str(message))
    #             self.messages.append(message)
    #
    #         self.messages = self.bm25_retrieve(self.messages)
    #
    #
    #     else:
    #         print(self.memory_path,' does not exist!')

    def __init__(self, memory_path, target, top_k, deadline, llm):
        super().__init__()
        self.memory_path = memory_path
        self.target = target
        self.top_k = top_k
        self.deadline = deadline
        self.model = llm

        # load the historical tweets / comments of the user
        if self.memory_path is not None and os.path.exists(self.memory_path):
            print('load ', self.memory_path)
            df = open(self.memory_path, 'r', errors='ignore').readlines()
            content_set = set()

            for d in df:
                try:
                    d = json.loads(d)
                except json.JSONDecodeError:
                    # 这一行 JSON 不合法，跳过
                    continue

                full_content = None

                # 1）tweet 结构：title + content（沿用你现在的逻辑）
                if "title" in d and "content" in d:
                    title = d.get("title", "").strip()
                    content = d.get("content", "").strip()
                    title = title.replace("\n", " ")
                    content = content.replace("\n", " ")

                    if title and content:
                        full_content = f"{title}\n{content}"
                    elif title:
                        full_content = title
                    elif content:
                        full_content = content
                    else:
                        # title 和 content 都空，跳过
                        continue

                # 2）旧格式 tweet：rawContent（向后兼容）
                elif "rawContent" in d:
                    full_content = d.get("rawContent", "").strip()

                # 3）reddit 评论：comment
                elif "comment" in d:
                    comment = d.get("comment", "").strip()
                    if not comment:
                        continue

                    # 3.1 过滤“纯链接”的评论（只包含 URL 或 URL + 少量空白）
                    # 比如 "https://xxx" 或 "https://xxx\n\nhttps://yyy"
                    # 思路：把换行替换成空格后，看是否只剩下若干个 url token
                    comment_flat = re.sub(r"\s+", " ", comment).strip()
                    # 如果所有 token 都是 url，就认为是纯链接评论
                    tokens = comment_flat.split()
                    url_pattern = re.compile(r"^https?://\S+$")
                    if tokens and all(url_pattern.match(t) for t in tokens):
                        # 纯链接评论，直接丢弃
                        continue

                    # 3.2 给评论加前缀，帮助 LLM 学习这是评论语料
                    full_content = "[Reddit Comment] " + comment

                # 4）既不是 tweet 也不是 comment，跳过
                else:
                    continue

                # 到这里 full_content 一定是非空字符串
                full_content = re.sub(r"\n+", "\n", full_content)

                # 5）去重和长度过滤（避免极短/重复内容）
                if full_content in content_set or len(full_content.split()) < 10:
                    continue
                content_set.add(full_content)

                # 6）时间 & 用户过滤（保持原有逻辑；如果 date / user 字段缺失就跳过）
                if "date" not in d or "user" not in d or "username" not in d["user"]:
                    continue

                post_time = d["date"][:19]  # 注意你的 date 格式是 "YYYY/MM/DD HH:MM:SS" 的话，这里只是截取前 19 位
                if self.deadline is not None and post_time > self.deadline:
                    # 如果你之后把 deadline 换成 datetime，可以在这里改比较方式
                    continue

                sender = d["user"]["username"]
                # 文件名去掉后缀当作用户名：.../Acrzyguy.txt -> Acrzyguy
                if sender != self.memory_path.split('/')[-1][:-4]:
                    continue

                message = TwitterMessage(content=full_content, post_time=post_time, sender=sender)
                self.messages.append(message)

            # 用 BM25 做一次筛选，保持和原实现一致
            self.messages = self.bm25_retrieve(self.messages)
        else:
            print(self.memory_path, ' does not exist!')

    def add_message(self, messages: List[Message]) -> None:
        for message in messages:
            self.messages.append(message)

    def reset(self) -> None:
        self.messages = []

    def bm25_retrieve(self, messages):
        if len(messages)==0:return []
        texts = [message.content for message in self.messages]
        # print("texts",texts)
        query = self.RETRIVEAL_QUERY.format(target=self.target)
        # print("query",query)
        bm25_retriever = build_bm25_retriever(texts)

        _, idx = bm25_retrieve_facts(bm25_retriever, texts, query, self.top_k)
        messages = [messages[i] for i in idx]
        # print("messages",messages)
        return messages
 

    async def summarize(self):
        self.has_summary = True
        messages = self.messages
        print('summarize personal experience:', len(messages))
        if len(messages)==0:
            self.summary=''
            return
        texts = self.to_string(add_sender_prefix=True)
        prompt = self.SUMMARIZATION_PROMPT.format(
            new_events=texts
        )
        response = await openai.ChatCompletion.acreate(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_summary_length,
            temperature=0.5,
        )
        self.summary =  response["choices"][0]["message"]["content"]   
        message = Message(content=self.summary)
        self.add_message([message])
        
    def to_string(self, add_sender_prefix: bool = False) -> str:   
        if add_sender_prefix:
            return "\n".join(
                [
                    f"[{message.sender}] posted a tweet: {message.content}"
                    if message.sender != ""
                    else message.content
                    for message in self.messages
                ]
            )
        else:
            return "\n".join([message.content for message in self.messages])