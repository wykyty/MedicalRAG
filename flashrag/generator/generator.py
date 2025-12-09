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


class EncoderDecoderGenerator(BaseGenerator):
    """Class for encoder-decoder model"""

    def __init__(self, config):
        super().__init__(config)
        model_config = AutoConfig.from_pretrained(self.model_path)
        arch = model_config.architectures[0].lower()
        if "t5" in arch or 'fusionindecoder' in arch:
            if self.fid:
                from flashrag.generator.fid import FiDT5
                self.model = FiDT5.from_pretrained(self.model_path)
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        else:
            if self.fid:
                assert False, "FiD only support T5"
            self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    def update_additional_setting(self):
        self.fid = self._config["use_fid"]

    def encode_passages(self, batch_text_passages: List[List[str]]):
        import torch
        # need size: [batch_size, passage_num, passage_len]
        passage_ids, passage_masks = [], []
        for text_passages in batch_text_passages:
            p = self.tokenizer(
                text_passages,
                max_length=self.max_input_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids, passage_masks.bool()

    def generate(self, input_list: List, batch_size=None, **params):
        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        responses = []
        for idx in trange(0, len(input_list), batch_size, desc="Generation process: "):
            batched_prompts = input_list[idx : idx + batch_size]
            if self.fid:
                # assume each input in input_list is a list, contains K string
                input_ids, attention_mask = self.encode_passages(batched_prompts)
                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
            else:
                inputs = self.tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_len,
                ).to(self.device)

            # TODO: multi-gpu inference
            import torch
            with torch.inference_mode():
                if self.fid:
                    if 'max_new_tokens' in generation_params:
                        max_new_tokens = generation_params.pop('max_new_tokens')
                    else:
                        max_new_tokens = 32

                    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.pad_token_id)
                else:
                    outputs = self.model.generate(**inputs, **generation_params)
            outputs = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            responses += outputs

        return responses


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

    #     # === 新增：Gemma 停止 token 兜底（SamplingParams 之前）===
    #     if "gemma" in self.model_name.lower():
    #         extra_stop_ids = []

    #     # 1) 优先尝试 Gemma 的 <end_of_turn>（部分权重/分支可能没有）
    #     try:
    #         eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    #         if isinstance(eot_id, int) and eot_id >= 0:
    #             extra_stop_ids.append(int(eot_id))
    #     except Exception:
    #         pass

    #     # 2) 兜底用模型自带的 eos_token_id
    #     eos_id = getattr(self.tokenizer, "eos_token_id", None)
    #     if isinstance(eos_id, int) and eos_id >= 0:
    #         extra_stop_ids.append(int(eos_id))

    # # 合并到 vLLM 支持的参数：stop_token_ids（注意：不是 eos_token_id）
    #     if extra_stop_ids:
    #         old_ids = generation_params.get("stop_token_ids", None)
    #         if old_ids is None:
    #             merged = extra_stop_ids
    #         else:
    #             merged = list({int(x) for x in list(old_ids) + extra_stop_ids})
    #         generation_params["stop_token_ids"] = merged



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



# import requests
# import numpy as np
# from copy import deepcopy
# from typing import List

# class HostGenerator(BaseGenerator):
#     """Class for generator that calls external vllm http server."""

#     def __init__(self, config):
#         super().__init__(config)

#     def update_additional_setting(self):
#         if "api_url" not in self._config:
#             self.api_url = "http://localhost:8000/v1/completions"
#         else:
#             self.api_url = self._config["api_url"]


#     def generate(
#         self,
#         input_list: List[str],
#         return_raw_output=False,
#         return_scores=False,
#         **params,
#     ):
#         """
#         仿照 Multi-Answer-Code/inference_folder/utils/llm_local.py process_batch_llm_host_prompts_log 
#         的调用和处理方式来改写，不再使用 generation_params，直接使用 HostGenerator 配置里的参数。
#         增加进度条显示。
#         """
#         from tqdm import tqdm

#         # 支持单条字符串输入
#         if isinstance(input_list, str):
#             input_list = [input_list]

#         # 用于批量发送和进度条显示
#         results = []
#         idxs = list(range(len(input_list)))

#         batch_size = params.get("batch_size", 512)

#         # 收集所有结果
#         result_list = []
#         for batch_start in tqdm(range(0, len(input_list), batch_size), desc="HostGenerator Progress"):
#             batch_end = min(batch_start + batch_size, len(input_list))
#             batch_inputs = input_list[batch_start:batch_end]
#             payload = {
#                 "prompt": batch_inputs,
#                 "temperature": params.get("temperature", 0),
#                 "top_p": params.get("top_p", 1.0),
#                 # "max_tokens": params.get("max_tokens", 4096),  # 加上这行试试效果
#                 # 结果：加上这样vllm报错, 400 Client Error
#                 "logprobs": params.get("logprobs", 20 if return_scores else None),
#                 "stream": False,
#             }
#             payload = {k: v for k, v in payload.items() if v is not None}

#             try:
#                 resp = requests.post(self.api_url, json=payload, timeout=180)
#                 resp.raise_for_status()
#                 data = resp.json()
#                 batch_results = data.get("outputs", data.get("choices", []))   # 支持vllm/vllm-api格式
#                 result_list.extend(batch_results)
#             except Exception as e:
#                 raise RuntimeError(f"Error in VLLM host call: {str(e)}")

#         # raw output
#         if return_raw_output:
#             return result_list

#         '''
#         check llm返回的answer
#         '''
#         # === 保存 gpt 输出 ===
#         import json
#         output_file = "/data0/wyh/RAG-Safer/src/retrieved_doc_wyk/gpt_output.json"
#         try:
#             with open(output_file, "w", encoding="utf-8") as fout:
#                 json.dump(result_list, fout, ensure_ascii=False, indent=2)
#             print(f"✅ GPT输出已保存到: {output_file}")
#         except Exception as e:
#             print(f"❌ 保存 GPT 输出时出错: {e}")
            

#         # 收集文本输出
#         generated_texts = []
#         for candidate in result_list:
#             # 仿照FileContext_0，如果有text字段直接用，否则看outputs（标准vllm格式），否则补空
#             if isinstance(candidate, list):
#                 # batch老格式
#                 generated_texts.append([out.get("text", "") for out in candidate])
#             elif "text" in candidate:
#                 generated_texts.append(candidate.get("text", ""))
#             elif "outputs" in candidate:
#                 outputs = candidate["outputs"]
#                 if isinstance(outputs, list) and len(outputs) > 1:
#                     generated_texts.append([out.get("text", "") for out in outputs])
#                 elif isinstance(outputs, list) and len(outputs) == 1:
#                     generated_texts.append(outputs[0].get("text", ""))
#                 else:
#                     generated_texts.append("")
#             else:
#                 generated_texts.append("")
#         base_output = generated_texts

#         if return_scores:
#             scores = []
#             for candidate in result_list:
#                 output_scores = []
#                 # 对应标准格式，兼容老格式
#                 if "outputs" in candidate:
#                     outputs = candidate["outputs"]
#                 elif isinstance(candidate, list):
#                     outputs = candidate
#                 else:
#                     outputs = [candidate]
#                 for single_out in outputs:
#                     logprobs_val = single_out.get("logprobs", None)
#                     if logprobs_val:
#                         token_probs = [np.exp(lp) if lp is not None else np.nan for lp in logprobs_val]
#                         output_scores.append(token_probs)
#                     else:
#                         output_scores.append([])
#                 if len(output_scores) == 1:
#                     scores.append(output_scores[0])
#                 else:
#                     scores.append(output_scores)
#             return base_output, scores
#         else:
#             return base_output


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
        super().__init__(config)  # Initialize BaseGenerator
        # 配置API参数
        api_setting = config["api_setting"] if "api_setting" in config else {}
        self.api_key = api_setting.get("api_key")  
        self.api_url = api_setting.get("api_url")  
        self.model = api_setting.get("model_name", "gpt-4o-mini")  # Default model
        self.temperature = api_setting.get("temperature", 0.0)
        self.top_p = api_setting.get("top_p", 1.0)
        self.concurrency = api_setting.get("concurrency", 500)
        self.max_retries = api_setting.get("max_retries", 3)
        self.system_prompt = ("You are a helpful assistant.")  # Default system prompt

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




class HFCausalLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on hf."""

    def __init__(self, config, model=None):
        super().__init__(config)
        self.model, self.tokenizer = self._load_model(model=model)
        if self.lora_path is not None:
            self.use_lora = True
            self.model.load_adapter(self.lora_path)

    def update_additional_setting(self):
        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model.to(self.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if "qwen" not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    def add_new_tokens(self, token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]"):
        import torch
        del self.model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        # get original embedding weight matrix
        embedding_layer = self.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight
        original_vocab_size, embedding_dim = embedding_weights.shape

        new_tokens_weights = torch.load(token_embedding_path)
        new_tokens_length = new_tokens_weights.shape[0]

        # expand vocabulary
        new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
        self.tokenizer.add_tokens(new_tokens)

        # create new embedding matrix
        new_vocab_size = original_vocab_size + new_tokens_length
        new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim)

        # copy original embeddings to the new weights
        new_embedding_weights[:original_vocab_size, :] = embedding_weights

        # append virtual token embeddings to the new weights
        for token, embedding in zip(new_tokens, new_tokens_weights):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            new_embedding_weights[token_id] = embedding

        # update the embedding table
        # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
        embedding_layer.weight.data = new_embedding_weights
        self.model.eval()
        self.model.cuda()

    def generate(
        self,
        input_list: List[str],
        batch_size=None,
        return_scores=False,
        return_dict=False,
        **params,
    ):
        """Generate batches one by one. The generated content needs to exclude input."""

        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        # set eos token for llama
        if "llama" in self.model_name.lower():
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if "eos_token_id" in generation_params:
                generation_params["eos_token_id"].extend(extra_eos_tokens)
            else:
                generation_params["eos_token_id"] = extra_eos_tokens

        responses = []
        scores = []
        generated_token_ids = []
        generated_token_logits = []

        import torch
        for idx in trange(0, len(input_list), batch_size, desc="Generation process: "):
            with torch.inference_mode():
                torch.cuda.empty_cache()
                batched_prompts = input_list[idx : idx + batch_size]
                inputs = self.tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_len,
                ).to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_params,
                )

                generated_ids = outputs.sequences
                logits = torch.stack(outputs.scores, dim=1).softmax(-1)
                generated_ids = generated_ids[:, inputs["input_ids"].shape[-1] :]
                gen_score = torch.gather(logits, 2, generated_ids[:, :, None]).squeeze(-1).cpu().tolist()
                scores.extend(gen_score)

            # get additinoal info
            if return_dict:
                batch_generated_token_ids = generated_ids.detach().cpu()
                batch_generated_token_logits = (
                    torch.cat(
                        [token_scores.unsqueeze(1) for token_scores in outputs.scores],
                        dim=1,
                    )
                    .detach()
                    .cpu()
                )
                if batch_generated_token_ids.shape[1] < generation_params["max_new_tokens"]:
                    real_batch_size, num_generated_tokens = batch_generated_token_ids.shape
                    padding_length = generation_params["max_new_tokens"] - num_generated_tokens
                    padding_token_ids = torch.zeros(
                        (real_batch_size, padding_length),
                        dtype=batch_generated_token_ids.dtype,
                    ).fill_(self.tokenizer.pad_token_id)
                    padding_token_logits = torch.zeros(
                        (
                            real_batch_size,
                            padding_length,
                            batch_generated_token_logits.shape[-1],
                        ),
                        dtype=batch_generated_token_logits.dtype,
                    )
                    batch_generated_token_ids = torch.cat([batch_generated_token_ids, padding_token_ids], dim=1)
                    batch_generated_token_logits = torch.cat(
                        [batch_generated_token_logits, padding_token_logits],
                        dim=1,
                    )
                generated_token_ids.append(batch_generated_token_ids)
                generated_token_logits.append(batch_generated_token_logits)

            for i, generated_sequence in enumerate(outputs.sequences):
                input_ids = inputs["input_ids"][i]
                text = self.tokenizer.decode(
                    generated_sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                new_text = text[prompt_length:]

                if stop_sym is not None:
                    strip_stopword = True
                    # Find the first occurrence of any stop word
                    lower_stop_index = len(new_text)  # Default to end of text
                    for sym in stop_sym:
                        stop_index = new_text.find(sym)
                        if stop_index != -1:
                            # Adjust stop index based on whether we're stripping the stop word
                            stop_index += 0 if strip_stopword else len(sym)
                            lower_stop_index = min(stop_index, lower_stop_index)

                    # Cut the text at the first stop word found (if any)
                    new_text = new_text[:lower_stop_index]

                responses.append(new_text.strip())

        if return_dict:
            generated_token_ids = torch.cat(generated_token_ids, dim=0)
            generated_token_logits = torch.cat(generated_token_logits, dim=0)
            return {
                "generated_token_ids": generated_token_ids,
                "generated_token_logits": generated_token_logits,
                "responses": responses,
                "scores": scores,
            }

        if return_scores:
            return responses, scores
        else:
            return responses


    def cal_gen_probs(self, prev, next):
        import torch
        input_ids = self.tokenizer.encode(prev, add_special_tokens=False)
        target_ids = self.tokenizer.encode(next, add_special_tokens=False)
        context_ids = input_ids + target_ids
        context_tensor = torch.tensor([context_ids]).to(self.device)
        with torch.inference_mode():
            outputs = self.model(context_tensor)
            logits = outputs.logits
            logits = logits[0, len(input_ids) - 1 : len(context_ids) - 1, :]
            logits = logits.to(torch.float32).detach().cpu()
            # softmax to normalize
            probs = torch.softmax(logits, dim=-1)
            # obtain probs of target_ids
            target_probs = probs[range(len(target_ids)), target_ids].numpy()

        return logits, target_probs


class FastChatGenerator(HFCausalLMGenerator):
    def __init__(self, config, model=None):
        super().__init__(config)

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""

        def get_gpu_memory():
            """Get available memory for each GPU."""
            import torch
            gpu_memory = []
            for gpu_id in range(self.gpu_num):
                with torch.cuda.device(gpu_id):
                    device = torch.cuda.current_device()
                    gpu_properties = torch.cuda.get_device_properties(device)
                    total_memory = gpu_properties.total_memory / (1024**3)
                    allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                    available_memory = total_memory - allocated_memory
                    gpu_memory.append(available_memory)
            return gpu_memory

        if model is None:
            from fastchat.model import load_model

            if "gpu_memory_utilization" not in self._config:
                gpu_memory_utilization = 0.85
            else:
                gpu_memory_utilization = self._config["gpu_memory_utilization"]
            max_gpu_memory = None
            import torch
            self.gpu_num = torch.cuda.device_count()
            if self.gpu_num > 1:
                available_gpu_memory = get_gpu_memory()
                max_gpu_memory = str(int(min(available_gpu_memory) * gpu_memory_utilization)) + "GiB"

            model, tokenizer = load_model(
                self.model_path,
                device=get_device(),
                num_gpus=self.gpu_num,
                max_gpu_memory=max_gpu_memory,
                load_8bit=False,
                cpu_offloading=False,
                debug=False,
            )

        else:
            model.cuda()
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if "qwen" not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer


# ——————————————————————————————————————————————————————————————————————
# 并行版本

# 并行
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
                "/data0/wyh/RAG-Safer/src/retrieved_doc_wyk/gpt_output.json")

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

