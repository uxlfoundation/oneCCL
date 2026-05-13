#!/usr/bin/env python3

# Copyright 2016-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM inference tests with unittest/xmlrunner compatibility"""

import os
import sys
import time
import unittest
import pytest
from vllm import LLM, SamplingParams


# Global LLM instance for unittest tests
_llm_instance = None


def get_llm_instance():
    """Get or initialize the shared LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        model_name = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
        tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "2"))

        print(f"\nInitializing vLLM model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")

        _llm_instance = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="float16"
        )

        print("Model loaded successfully")

        # Warm-up run
        print("Running warm-up...")
        messages = [{"role": "user", "content": "Hi"}]
        _ = _llm_instance.chat(messages, sampling_params=SamplingParams(max_tokens=10, temperature=0.7))
        print("Warm-up complete\n")

    return _llm_instance


# Pytest fixture to initialize the model once
@pytest.fixture(scope="module")
def llm_model():
    """Initialize vLLM model once for all tests"""
    return get_llm_instance()


def run_inference_test(llm, prompt, expected_keywords=None, min_tokens=10, max_tokens=128):
    """Helper function to run inference and validate output"""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt}
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
    )

    start_time = time.perf_counter()
    outputs = llm.chat(messages, sampling_params=sampling_params)
    end_time = time.perf_counter()

    output = outputs[0]
    generated_text = output.outputs[0].text
    prompt_tokens = len(output.prompt_token_ids)
    output_tokens = len(output.outputs[0].token_ids)
    total_time = end_time - start_time
    tokens_per_sec = output_tokens / total_time

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}...")
    print(f"Tokens: {output_tokens}, Time: {total_time:.2f}s, Throughput: {tokens_per_sec:.2f} tok/s")

    # Validations
    assert len(generated_text) > 0, "Generated text is empty"
    assert output_tokens >= min_tokens, f"Output too short: {output_tokens} < {min_tokens}"
    assert tokens_per_sec > 0, "Invalid throughput"

    # Check for expected keywords if provided
    if expected_keywords:
        text_lower = generated_text.lower()
        found_keywords = [kw for kw in expected_keywords if kw.lower() in text_lower]
        assert len(found_keywords) > 0, f"None of expected keywords {expected_keywords} found in output"

    return {
        "text": generated_text,
        "tokens": output_tokens,
        "time": total_time,
        "throughput": tokens_per_sec
    }


class TestVLLMInference:
    """Test cases for vLLM inference"""
    def test_code_explanation(self, llm_model):
        """Test: Code explanation"""
        result = run_inference_test(
            llm_model,
            prompt="Explain what this Python code does: for i in range(10): print(i**2)",
            expected_keywords=["loop", "square", "print"],
            min_tokens=15
        )

    def test_throughput_benchmark(self, llm_model):
        """Test: Throughput is within reasonable range"""
        result = run_inference_test(
            llm_model,
            prompt="Write a detailed explanation of how Python's list comprehension works:",
            min_tokens=30,
            max_tokens=256
        )
        # Check that throughput is reasonable (e.g., > 5 tokens/sec)
        assert result["throughput"] > 5.0, f"Throughput too low: {result['throughput']:.2f} tok/s"


# Unittest wrapper for xmlrunner compatibility
class TestVLLMInferenceUnittest(unittest.TestCase):
    """Unittest-compatible test wrapper for xmlrunner"""

    @classmethod
    def setUpClass(cls):
        """Initialize model once for all unittest tests"""
        cls.llm = get_llm_instance()

    def test_code_explanation(self):
        """Test: Code explanation"""
        result = run_inference_test(
            self.llm,
            prompt="Explain what this Python code does: for i in range(10): print(i**2)",
            expected_keywords=["loop", "square", "print"],
            min_tokens=15
        )

    def test_throughput_benchmark(self):
        """Test: Throughput is within reasonable range"""
        result = run_inference_test(
            self.llm,
            prompt="Write a detailed explanation of how Python's list comprehension works:",
            min_tokens=30,
            max_tokens=256
        )
        # Check that throughput is reasonable (e.g., > 5 tokens/sec)
        self.assertGreater(result["throughput"], 5.0, f"Throughput too low: {result['throughput']:.2f} tok/s")


if __name__ == "__main__":
    # Support both unittest/xmlrunner and direct execution
    if 'xmlrunner' in sys.modules or '--xmlrunner' in sys.argv:
        import xmlrunner
        # Get output filename from environment or use default
        output_file = os.environ.get('XML_OUTPUT_FILE', 'test-reports')
        runner = xmlrunner.XMLTestRunner(output=output_file, verbosity=2)
        unittest.main(testRunner=runner, argv=[sys.argv[0]])
    else:
        # Use unittest for direct execution
        unittest.main(verbosity=2)
