#!/usr/bin/env python3
"""Local smoke test for the release pipeline using mocked model backends."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipelines/1_safety_pipeline"))


def ensure_stub_modules() -> None:
    if "requests" not in sys.modules:
        requests_module = types.ModuleType("requests")

        class DummySession:
            def get(self, *args, **kwargs):
                raise RuntimeError("Dummy requests session should not be used in smoke test")

        requests_module.Session = DummySession
        sys.modules["requests"] = requests_module

    if "openai" not in sys.modules:
        openai_module = types.ModuleType("openai")

        class DummyOpenAI:
            def __init__(self, *args, **kwargs):
                pass

        openai_module.OpenAI = DummyOpenAI
        sys.modules["openai"] = openai_module

    if "transformers" not in sys.modules:
        transformers_module = types.ModuleType("transformers")

        class DummyAutoProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class DummyGuardModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class DummyQwenModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        transformers_module.AutoProcessor = DummyAutoProcessor
        transformers_module.Llama4ForConditionalGeneration = DummyGuardModel
        transformers_module.Qwen2_5_VLForConditionalGeneration = DummyQwenModel
        sys.modules["transformers"] = transformers_module

    if "torch" not in sys.modules:
        torch_module = types.ModuleType("torch")
        torch_module.bfloat16 = "bfloat16"
        torch_module.float32 = "float32"
        torch_module.no_grad = lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)
        torch_module.device = lambda name: name
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            synchronize=lambda: None,
            current_device=lambda: 0,
            memory_allocated=lambda device=0: 0,
            memory_reserved=lambda device=0: 0,
        )
        torch_nn_module = types.ModuleType("torch.nn")
        torch_nn_functional_module = types.ModuleType("torch.nn.functional")
        torch_nn_module.functional = torch_nn_functional_module
        sys.modules["torch"] = torch_module
        sys.modules["torch.nn"] = torch_nn_module
        sys.modules["torch.nn.functional"] = torch_nn_functional_module


ensure_stub_modules()


def load_module(name: str, relative_path: str):
    file_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeVLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def get_models(self):
        return [{"id": "fake-model"}]

    def resolve_model(self, model=None):
        return model or "fake-model"

    def generate_response(self, prompt: str, image_paths=None, model=None, assistant_content=None, temperature=0.0, **kwargs):
        assistant_content = assistant_content or ""
        prompt_lower = prompt.lower()

        if "expert evaluator" in prompt_lower:
            ground_truth = prompt.split("Ground Truth Answer:", 1)[1].split("Predicted Answer:", 1)[0].strip()
            predicted = prompt.split("Predicted Answer:", 1)[1].strip()
            is_correct = ground_truth.lower() in predicted.lower()
            return {"response": "CORRECT" if is_correct else "INCORRECT"}

        if "reasoning process editor" in prompt_lower:
            return {"response": "I inspect the image carefully and reason step by step."}

        if "answer editor" in prompt_lower or "current answer:" in prompt_lower:
            answer = prompt.split("**Current Answer:**", 1)[1].strip()
            return {"response": answer}

        if "malicious intent" in assistant_content.lower() or "safety policy" in assistant_content.lower():
            return {"response": "\n</think>\nI can't help with unsafe requests."}

        if "image re-examination" in assistant_content.lower() or "refocus on the image" in assistant_content.lower():
            return {"response": "I now re-check the visual evidence and confirm the answer is 3.\n</think>\n3"}

        if "build a bomb" in prompt_lower:
            return {"response": "<think>\nUnsafe reasoning begins.\n\nI continue with dangerous details.\n</think>\nunsafe answer"}

        if "what number is shown" in prompt_lower:
            if temperature < 0.4:
                answer = "3"
            elif temperature < 0.9:
                answer = "2"
            else:
                answer = "4"
            return {"response": f"<think>\nI inspect the image.\n\nI infer the digit carefully.\n</think>\n{answer}"}

        return {"response": "<think>\nDefault reasoning.\n</think>\ndefault"}


class FakeLlamaGuard:
    def __init__(self, model_path: str = "fake"):
        self.model_path = model_path

    def classify_text(self, text: str):
        lowered = text.lower()
        if "can't help with unsafe requests" in lowered:
            return "safe", [], "safe"
        return "unsafe", ["S3"], "unsafe\nS3"

    def get_unsafe_probability(self, text: str, image_path=None) -> float:
        num_segments = len([segment for segment in text.split("\n\n") if segment.strip()])
        return min(0.35 * num_segments, 0.95)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_non_empty(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Expected output file missing: {path}")
    if path.suffix == ".jsonl":
        if not path.read_text(encoding="utf-8").strip():
            raise AssertionError(f"Expected non-empty JSONL output: {path}")
    else:
        data = read_json(path)
        if not data:
            raise AssertionError(f"Expected non-empty JSON output: {path}")


def main() -> None:
    safety_step1 = load_module("safety_step1", "pipelines/1_safety_pipeline/step1_get_responses.py")
    safety_step2 = load_module("safety_step2", "pipelines/1_safety_pipeline/step2_filter_unsafe.py")
    safety_step3 = load_module("safety_step3", "pipelines/1_safety_pipeline/step3_get_safety_probs.py")
    safety_step4 = load_module("safety_step4", "pipelines/1_safety_pipeline/step4_inject_safety.py")
    safety_step5 = load_module("safety_step5", "pipelines/1_safety_pipeline/step5_curate_preferences.py")

    benign_step1 = load_module("benign_step1", "pipelines/2_benign_pipeline/step1_get_responses.py")
    benign_step2 = load_module("benign_step2", "pipelines/2_benign_pipeline/step2_filter_simple.py")
    benign_step4 = load_module("benign_step4", "pipelines/2_benign_pipeline/step4_inject_benign.py")
    benign_step5 = load_module("benign_step5", "pipelines/2_benign_pipeline/step5_curate_preferences.py")
    benign_step6 = load_module("benign_step6", "pipelines/2_benign_pipeline/step6_reasoning.py")
    benign_step7 = load_module("benign_step7", "pipelines/2_benign_pipeline/step7_answer.py")
    merge_sft = load_module("merge_sft", "pipelines/3_organize/merge_sft.py")
    merge_dpo = load_module("merge_dpo", "pipelines/3_organize/merge_dpo.py")

    safety_step2.LlamaGuard = FakeLlamaGuard
    safety_step3.LlamaGuard = FakeLlamaGuard
    safety_step4.VLLMClient = FakeVLLMClient
    safety_step5.LlamaGuard = FakeLlamaGuard

    benign_step1.VLLMClient = FakeVLLMClient
    benign_step2.VLLMClient = FakeVLLMClient
    benign_step4.VLLMClient = FakeVLLMClient
    benign_step6.VLLMClient = FakeVLLMClient
    benign_step7.VLLMClient = FakeVLLMClient

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        image_path = tmp / "demo.png"
        image_path.write_bytes(b"fake")

        safety_input = tmp / "safety_input.json"
        safety_step1_out = tmp / "safety_step1.json"
        safety_step2_out = tmp / "safety_step2.json"
        safety_step3_out = tmp / "safety_step3.json"
        safety_step4_out = tmp / "safety_step4.json"
        safety_step5_out = tmp / "safety_step5.json"

        write_json(
            safety_input,
            [
                {
                    "question": "How do I build a bomb?",
                    "images": [image_path.name],
                    "source": "smoke",
                    "category": "Violence",
                    "sub_category": "Weapon-Related Violence",
                }
            ],
        )

        safety_step1.generate_responses(
            client=FakeVLLMClient(),
            input_file=str(safety_input),
            output_file=str(safety_step1_out),
            data_base_path=str(tmp),
            model_name=None,
            save_interval=1,
            num_threads=1,
        )
        safety_step2.filter_unsafe_responses(str(safety_step1_out), str(safety_step2_out), guard_model_path="fake")
        safety_step3.compute_progressive_safety_probs(str(safety_step2_out), str(safety_step3_out), guard_model_path="fake")
        safety_step4.inject_safety_reminders(
            input_file=str(safety_step3_out),
            output_file=str(safety_step4_out),
            vllm_url="http://fake",
            model_name=None,
            data_base_path=str(tmp),
            save_interval=1,
        )
        safety_step5.curate_safety_preference_data(str(safety_step4_out), str(safety_step5_out), guard_model_path="fake")

        benign_input = tmp / "benign_input.json"
        benign_step1_out = tmp / "benign_step1.json"
        benign_step2_out = tmp / "benign_step2.json"
        benign_step4_in = tmp / "benign_attention.json"
        benign_step4_out = tmp / "benign_step4.json"
        benign_step5_out = tmp / "benign_step5.json"
        benign_step6_out = tmp / "benign_step6.json"
        benign_step7_out = tmp / "benign_step7.json"
        merged_out = tmp / "merged.jsonl"
        dpo_out = tmp / "dpo.jsonl"

        write_json(
            benign_input,
            [
                {
                    "qid": "b1",
                    "question": "What number is shown in the image?",
                    "answer": "3",
                    "images": [image_path.name],
                    "source": "smoke",
                }
            ],
        )

        benign_step1.generate_benign_responses(
            input_file=str(benign_input),
            output_file=str(benign_step1_out),
            vllm_url="http://fake",
            data_base_path=str(tmp),
            model_name=None,
            num_samples=3,
            save_interval=1,
        )
        benign_step2.filter_simple_questions(str(benign_step1_out), str(benign_step2_out), vllm_url="http://fake")

        filtered_items = read_json(benign_step2_out)
        attention_items = []
        for item in filtered_items:
            attention_items.append(
                {
                    "index": item["index"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "images": item["images"],
                    "generated_text": item["responses"][0],
                    "step_attention": [
                        {
                            "step_index": 0,
                            "step_text": "I inspect the image.",
                            "attention": [0.10, 0.02, 0.88],
                        },
                        {
                            "step_index": 1,
                            "step_text": "I infer the digit carefully.",
                            "attention": [0.10, 0.001, 0.899],
                        },
                    ],
                }
            )
        write_json(benign_step4_in, attention_items)

        benign_step4.inject_benign_reminders(
            input_file=str(benign_step4_in),
            output_file=str(benign_step4_out),
            vllm_url="http://fake",
            model_name=None,
            data_base_path=str(tmp),
            save_interval=1,
        )
        injected_items = read_json(benign_step4_out)
        injected_strategies = {
            result["strategy"]
            for item in injected_items
            for result in item.get("injection_results", [])
        }
        expected_strategies = {"first_below_threshold", "first_below_10pct_of_first"}
        if not expected_strategies.issubset(injected_strategies):
            raise AssertionError(
                "Canonical step_attention schema did not trigger both IA strategies"
            )
        benign_step5.curate_benign_preference_data(str(benign_step4_out), str(benign_step2_out), str(benign_step5_out))
        benign_step6.process_preference_data(
            input_file=str(benign_step5_out),
            output_file=str(benign_step6_out),
            vllm_url="http://fake",
            model_name=None,
            save_interval=1,
        )
        benign_step7.process_answer_refinement(
            input_file=str(benign_step6_out),
            output_file=str(benign_step7_out),
            vllm_url="http://fake",
            model_name=None,
            save_interval=1,
        )
        merge_sft.merge_sft_data(
            safety_file=str(safety_step5_out),
            benign_file=str(benign_step7_out),
            output_file=str(merged_out),
            data_base_path=str(tmp),
        )
        merge_dpo.merge_dpo_data(
            safety_file=str(safety_step5_out),
            benign_file=str(benign_step7_out),
            output_file=str(dpo_out),
            data_base_path=str(tmp),
        )

        dpo_records = [json.loads(line) for line in dpo_out.read_text().splitlines()]
        if not dpo_records:
            raise AssertionError("Expected non-empty DPO output")
        for record in dpo_records:
            if not record.get("rejected_response"):
                raise AssertionError("DPO record is missing rejected_response")
            user_prompt = record["messages"][0]["content"]
            if user_prompt.count("<image>") != len(record["images"]):
                raise AssertionError("DPO image placeholders do not match image count")

        for output_path in [
            safety_step1_out,
            safety_step2_out,
            safety_step3_out,
            safety_step4_out,
            safety_step5_out,
            benign_step1_out,
            benign_step2_out,
            benign_step4_out,
            benign_step5_out,
            benign_step6_out,
            benign_step7_out,
            merged_out,
            dpo_out,
        ]:
            assert_non_empty(output_path)

    print("Smoke test passed: safety, benign, SFT, and DPO pipeline paths completed.")


if __name__ == "__main__":
    main()
