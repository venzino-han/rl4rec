
from openai import OpenAI
import concurrent.futures

from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm
import os
import time
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer


class LLMGenerator:
    def __init__(self, args):
        self.args = args
        if args.use_deep_infra:
            deep_infra_api_key = os.getenv("DEEP_INFRA_API_KEY")
            self.deep_infra_client = OpenAI(
                # deep infra key from .env
                api_key=deep_infra_api_key,
                base_url="https://api.deepinfra.com/v1/openai",
            )

        else:
            tokenizer_name = args.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.llm = LLM(
                model=args.model_name, 
                tensor_parallel_size=args.num_gpus,
                max_model_len=args.max_input_tokens,
                max_num_batched_tokens=args.max_input_tokens,
                gpu_memory_utilization = args.gpu_memory_utilization,
                max_num_seqs=args.batch_size,
                quantization="awq_marlin" if "awq" in args.model_name.lower() else None,
                dtype=args.dtype,
                )
            print("Model loaded")
            self.sampling_params = SamplingParams(
                temperature=args.temperature,
                min_tokens=0,
                max_tokens=args.max_output_tokens,
                skip_special_tokens=False,
            )
            print("Sampling params set")

    def apply_chat_template(self, system_prompt, prompt):
        prompt = [
            {"role": "system", "content": system_prompt,},
            {"role": "user", "content": prompt,},
        ]
        return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    def generate_response(self, system_prompt, prompts):
        if self.args.use_deep_infra:
            # Deep Infra API 호출 및 응답 처리
            responses, token_nums = self._generate_response_with_deep_infra(system_prompt, prompts)
        else:
            prompts = [ self.apply_chat_template(system_prompt, prompt) for prompt in prompts]            
            print("Starting to generate responses with vLLM...")
            outputs = self.llm.generate(prompts, self.sampling_params)
            responses = [output.outputs[0].text for output in outputs]
            token_nums = [len(output.outputs[0].text) for output in outputs]
        return responses, token_nums

    def _generate_response_with_openai(self, prompt, system_prompt, model_name="gpt-4o-mini", max_tokens=512, temperature=0.1, timeout=20, max_retries=3):
        # Truncate prompts if they exceed the limit
        if len(prompt.split()) > 1024*12:
            prompt = " ".join(prompt.split()[-1024*12:])

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        result = None
        while result is None:
            try:
                completion = self.deep_infra_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout  # Timeout in seconds
                    )
                result = completion.choices[0].message.content
                return result
            except: 
                time.sleep(1)

        # If the request fails after max retries, raise an exception
        raise Exception(f"Request failed after {max_retries} retries due to timeout.")

    def _generate_responses_concurrently(
            self,
            prompts, system_prompt, model_name="gpt-4o-mini", max_tokens=512, 
            temperature=0.0, timeout=20, max_retries=3, max_concurrent_requests=8):
        """
        Sends multiple prompts to the OpenAI API concurrently and returns their results as a list, maintaining order.

        Args:
            prompts (list of str): List of prompts to process.
            system_prompt (str): System prompt to include in each request.
            model_name (str): The model to use for API requests.
            max_tokens (int): The maximum number of tokens for each response.
            temperature (float): Sampling temperature for the responses.
            timeout (int): Timeout for each API request.
            max_retries (int): Number of retries for a failed request.
            max_concurrent_requests (int): Maximum number of concurrent requests.

        Returns:
            list of str: List of responses corresponding to each prompt in the same order.
        """
        def send_request(index, prompt):
            """Helper function to send a single API request."""
            response = self._generate_response_with_openai(
                prompt=prompt,
                system_prompt=system_prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
            return index, response

        # Use ThreadPoolExecutor for concurrent requests
        results = ["None"] * len(prompts)  # Preallocate list for responses
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # Submit all prompts for processing with their indices
            future_to_index = {executor.submit(send_request, idx, prompt): idx for idx, prompt in enumerate(prompts)}

            # Collect the results as they are completed
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    index, response = future.result()
                    results[index] = response  # Store the result in the correct position
                except Exception as e:
                    print(f"Error processing prompt at index {future_to_index[future]} -> {e}")
                    results[future_to_index[future]] = "None"  # Append None for failed requests

        return results

    def _generate_response_with_deep_infra(self, system_prompt, prompts):
        # batch size 만큼 요청을 보내고 응답을 받아옴
        responses = []
        token_nums = []
        for i in tqdm(range(0, len(prompts), self.args.batch_size)):
            batch_prompts = prompts[i:i+self.args.batch_size]
            batch_responses = self._generate_responses_concurrently(
                batch_prompts, system_prompt, model_name=self.args.model_name, max_tokens=self.args.max_output_tokens,
                temperature=self.args.temperature, timeout=20, max_retries=3, max_concurrent_requests=self.args.batch_size
            )
            responses.extend(batch_responses)
            token_nums.extend([len(response.split())*2 for response in batch_responses])

            if i == 0:
                for j in range(4):
                    print(f"Example prompt:\n{batch_prompts[j]}\n---\n")
                    print(f"Example response:\n{batch_responses[j]}\n---\n")

        
        return responses, token_nums
