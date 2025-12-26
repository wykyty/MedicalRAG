from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoConfig,
)
from flashrag.generator.utils import resolve_max_tokens
from flashrag.utils import get_device


class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()
    
    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()
    def update_base_setting(self):
        self.model_name = self._config["generator_model"]
        self.model_path = self._config["generator_model_path"]

        self.max_input_len = self._config["generator_max_input_len"]
        self.batch_size = self._config["generator_batch_size"]
        self.device = self._config["device"]
        self.gpu_num = self._config['gpu_num']
        self.generation_params = self._config["generation_params"]
    
    def update_additional_setting(self):
        pass

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)
        # print('#' * 40)
        import inspect
        # print("[DEBUG] VLLMGenerator loaded from:", inspect.getsourcefile(type(self)), flush=True)
        from vllm import LLM
        if self.use_lora:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self.gpu_memory_utilization,
                enable_lora = True,
                max_lora_rank = 64,
                max_logprobs = 32016,
                max_model_len = self.max_model_len
            )
        else:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self.gpu_memory_utilization,
                max_logprobs = 32016,
                max_model_len = self.max_model_len
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        # print('#'*40)
        # print(f"self.gpu_num: {self.gpu_num}")
        # print('#'*40)
        # print(f"self.tensor_parallel_size: {self.tensor_parallel_size}")
        # print("#"*40)

        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self._config['generator_max_input_len']
        # print('*'*60)
        # print(self.max_input_len)
        # print(self.max_model_len)
        # print('*'*60)

    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        **params,
    ):
        from vllm import SamplingParams

        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0
        generation_params["seed"] = self._config["seed"]

        # handle param conflict
        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)

        # fix for llama3
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(input_list, sampling_params)

        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [
                [c.text for c in output.outputs] if len(output.outputs) > 1 else output.outputs[0].text
                for output in outputs
            ]
            base_output = generated_texts
        if return_scores:
            output_scores=[]
            scores = []
            for output in outputs:
                for single_output in output.outputs:
                    if single_output.logprobs:
                        token_probs = [np.exp(list(score_dict.values())[0].logprob) 
                                      for score_dict in single_output.logprobs]
                        output_scores.append(token_probs)
                    else:
                        output_scores.append([])
                if len(output_scores) == 1:
                    scores.append(output_scores[0])
                else:
                    scores.append(output_scores)
            return base_output, scores
        else:
            return base_output


"""
调用API进行生成
"""
import requests
import numpy as np
from tqdm import tqdm
import asyncio
import aiohttp
from copy import deepcopy
from typing import List
import random
from tqdm.asyncio import tqdm as tqdm_asyncio

class APIGenerator(BaseGenerator):
    """Class for generator that calls an external API server"""

    def __init__(self, config):
        super().__init__(config)  
        # 配置API参数
        api_setting = config["api_setting"] if "api_setting" in config else {}
        self.api_key = api_setting.get("api_key")  
        self.api_url = api_setting.get("api_url")  
        self.model = api_setting.get("generator_model")
        
        self.temperature = api_setting.get("temperature", 0.0)
        self.top_p = api_setting.get("top_p", 1.0)
        self.concurrency = api_setting.get("concurrency", 500)
        self.max_retries = api_setting.get("max_retries", 3)
        self.system_prompt = ("")  

    async def _call_api_batch(self, prompts: List[str]):
        """Call the custom API in batches using asynchronous requests."""
        semaphore = asyncio.Semaphore(self.concurrency)  # Control concurrency
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BatchEvalBot/1.0",
        }

        async def call_api(session, user_msg):
            """Single request logic with retry mechanism"""
            for attempt in range(self.max_retries):
                try:
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_msg}
                        ],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                    async with semaphore:
                        async with session.post(
                            self.api_url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=60),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                response = data.get("choices", [{}])[0].get("message", {}).get("content", None)
                                return response
                            else:
                                text = await resp.text()
                                print(f"⚠️ HTTP {resp.status} - attempt {attempt+1}: {text}")
                                if attempt < self.max_retries - 1:
                                    await asyncio.sleep((2 ** attempt) + random.random())
                except Exception as e:
                    print(f"❌ Exception in call_api (attempt {attempt+1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep((2 ** attempt) + random.random())
            return None

        async with aiohttp.ClientSession() as session:
            tasks = [call_api(session, prompt) for prompt in prompts]
            results = await tqdm_asyncio.gather(*tasks, desc="Generating with Custom API")
            return results

    def _process_judge_result(self, resp_str):
        """Process the response from the safety check"""
        if not resp_str:
            return False
        norm = str(resp_str).strip().lower()
        if "harmful" in norm or "unsafe" in norm or "toxic" in norm:
            return True
        return False

    def generate(
        self,
        input_list: List[str],
        return_raw_output=True,
        return_scores=False,
        **params,
    ):
        """Generate responses using the custom API"""
        if isinstance(input_list, str):
            input_list = [input_list]

        # Call the custom API in batches
        prompts = input_list
        responses = asyncio.run(self._call_api_batch(prompts))
        
        # Process responses
        result_list = []
        for response in responses:
            if response:
                result_list.append(response)
            else:
                result_list.append("")

        # raw output
        if return_raw_output:
            return result_list

        # If scores are needed
        if return_scores:
            scores = [self._process_judge_result(resp) for resp in responses]
            return result_list, scores



# 本地部署模型，然后进行生成
import aiohttp
import asyncio
import json
import numpy as np
from typing import List, Any
from tqdm.asyncio import tqdm as tqdm_async

class HostGenerator(BaseGenerator):
    """Async client for /v1/completions with controlled concurrency."""

    def __init__(self, config):
        super().__init__(config)

    def update_additional_setting(self):
        setting = self._config["generator_setting"] if "generator_setting" in self._config else {}
        self.api_url = setting.get("api_url", "http://localhost:8001/v1/completions")
        self.temperature = float(setting.get("temperature", 0.0))
        self.top_p = float(setting.get("top_p", 1.0))
        self.max_tokens = int(setting.get("max_tokens", 1024))   # 越小越快，按需设

        self.request_timeout = int(setting.get("request_timeout_sec", 600))
        self.default_logprobs = setting.get("logprobs", None)
        self.stop = setting.get("stop", None)

        self.concurrency = int(setting.get("concurrency", 8))
        self.save_path = setting.get("host_raw_save_path",
                "/data/wyh/MedicalRAG/output/gpt_output.json")

    def _build_payload(self, prompt: str, logprobs=None):
        if not isinstance(prompt, str):
            prompt = str(prompt)
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.stop:
            payload["stop"] = self.stop
        if logprobs is not None:
            payload["logprobs"] = int(logprobs)
        return payload

    @staticmethod
    def _parse_text(data: dict) -> str:
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text", "") or ""
        if "outputs" in data and data["outputs"]:
            return data["outputs"][0].get("text", "") or ""
        return ""

    @staticmethod
    def _parse_logprobs(data: dict):
        try:
            ch0 = (data.get("choices") or [{}])[0]
            lp = ch0.get("logprobs")
            if not lp:
                return []
            tlp = lp.get("token_logprobs")
            if tlp is None:
                return []
            return [float(np.exp(x)) if x is not None else None for x in tlp]
        except Exception:
            return []

    async def _worker(self, session: aiohttp.ClientSession, prompt: str, return_scores: bool):
        payload = self._build_payload(prompt, self.default_logprobs if return_scores else None)
        async with session.post(self.api_url, json=payload, timeout=self.request_timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
            data = await resp.json()
            text = self._parse_text(data)
            scores = self._parse_logprobs(data) if return_scores else None
            return data, text, scores

    async def _run_async(self, prompts, return_raw_output: bool, return_scores: bool):
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        sem = asyncio.Semaphore(self.concurrency)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i, p in enumerate(prompts):
                async def run_one(idx=i, prompt=p):
                    async with sem:
                        data, text, sc = await self._worker(session, prompt, return_scores)
                        return idx, data, text, sc
                tasks.append(run_one())

            n = len(tasks)
            raw_results = [None] * n
            texts       = [None] * n
            scores      = [None] * n if return_scores else None

            # 用 as_completed 做进度，但按 idx 回填
            for fut in tqdm_async.as_completed(tasks, desc="HostGenerator(async completions)"):
                idx, data, text, sc = await fut
                raw_results[idx] = data
                texts[idx]       = text
                if return_scores:
                    scores[idx] = sc

        if return_raw_output and return_scores:
            return raw_results, scores
        if return_raw_output:
            return raw_results
        if return_scores:
            return texts, scores
        return texts

        # # 保存完整响应
        # try:
        #     with open(self.save_path, "w", encoding="utf-8") as f:
        #         json.dump(raw_results, f, ensure_ascii=False, indent=2)
        #     print(f"✅ GPT原始输出已保存到: {self.save_path}")
        # except Exception as e:
        #     print(f"❌ 保存 GPT 输出时出错: {e}")


    def generate(self, input_list: List[str], return_raw_output=False, return_scores=False, **params):
        if isinstance(input_list, str):
            input_list = [input_list]
        # 允许动态覆盖少量参数
        self.max_tokens = int(params.get("max_tokens", self.max_tokens))
        self.temperature = float(params.get("temperature", self.temperature))
        self.top_p = float(params.get("top_p", self.top_p))
        self.stop = params.get("stop", self.stop)
        self.concurrency = int(params.get("concurrency", self.concurrency))

        return asyncio.run(self._run_async(input_list, return_raw_output, return_scores))

