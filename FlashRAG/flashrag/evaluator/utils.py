import re
import string
import random

from typing import List, Tuple, Dict, Any, Optional, Callable
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 获取 q-docs-list
def get_q_docs_list(data):
    # 准备question-docs
    question_list = getattr(data, "question", None)
    retrieval_results = getattr(data, "retrieval_result", None)
    if question_list is None or retrieval_results is None:
        raise ValueError("Data must have 'question' and 'retrieval_result' fields.")
    if len(question_list) != len(retrieval_results):
        raise ValueError("Length of 'question' and 'retrieval_result' must match.")

    # 构建列表
    q_docs_list = []
    for q, item in zip(question_list, retrieval_results):
        docs = []
        for it in item:
            contents = it.get("contents", None)
            if contents is None:
                raise ValueError("Each retrieval_result must have a 'contents' field.")

            # 支持 contents 是字符串或列表两种情况
            if isinstance(contents, str):
                docs.append(contents.strip())
            elif isinstance(contents, list):
                docs.append(c.strip() for c in contents if c.strip())
            else:
                raise ValueError("'contents' must be str or list of str.")

        q_docs_list.append((q, docs))
    return q_docs_list


# 通用的批量调用 GPT 接口函数
async def call_gpt_api_batch_common(
    prompts: List[str],
    build_messages: Callable[[str], List[Dict[str, str]]],
    *,
    api_key: Optional[str],
    api_url: str,
    model: str,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    request_timeout_sec: int,
    tqdm_desc: str,
) -> List[Optional[str]]:
    """
    通用的批量调用 GPT 接口函数。
    两个类只需要提供：
      - build_messages(user_msg) 函数
      - 自己的配置参数（api_url / model / 并发 / 超时 等）
    """

    semaphore = asyncio.Semaphore(concurrency)
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "Content-Type": "application/json",
        "User-Agent": "BatchEvalBot/1.0",
    }

    async def call_api(session: aiohttp.ClientSession, user_msg: str) -> Optional[str]:
        payload = {
            "model": model,
            "messages": build_messages(user_msg),
            "temperature": temperature,
            "top_p": top_p,
        }

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    async with session.post(
                        api_url,
                        json=payload,
                        headers=headers,
                    ) as resp:
                        text = await resp.text()
                        if resp.status == 200:
                            data = await resp.json()
                            response = data.get("choices", [{}])[0].get("message", {}).get("content")
                            return response

                        # 429 / 5xx 做指数退避
                        if resp.status in (429, 500, 502, 503, 504):
                            wait = (2 ** attempt) + random.random()
                            print(
                                f"↻ Retryable HTTP {resp.status} on attempt {attempt + 1}, "
                                f"wait {wait:.2f}s. Body: {text[:300]}"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(wait)
                                continue

                        # 其他状态码直接记录
                        print(f"⚠️ HTTP {resp.status} - attempt {attempt + 1}: {text[:300]}")

            except Exception as e:
                wait = (2 ** attempt) + random.random()
                print(f"❌ Exception attempt {attempt + 1}: {e}. Backoff {wait:.2f}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                    continue

        return None

    timeout = aiohttp.ClientTimeout(total=max(request_timeout_sec, 30))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [call_api(session, p) for p in prompts]
        results = await tqdm_asyncio.gather(*tasks, desc=tqdm_desc)
        return results