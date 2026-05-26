#!/usr/bin/env python3
"""Convert a HuggingFace chat model to .rkllm format for Rockchip NPUs.

Usage (on a CUDA PC — Vast.ai, Lambda, or local GPU workstation)::

    python3 build_rkllm_model.py \\
        --model Qwen/Qwen3-0.6B-Instruct \\
        --target RK3576 \\
        --quant W4A16

Output: ``qwen3-0.6b-instruct_W4A16_RK3576.rkllm``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile


def _calibration_data(model_name: str) -> list[dict[str, str]]:
    """20 mixed zh/en examples for quantisation calibration.

    The format is [{"input": "...", "target": "..."}, ...].
    Input lines use the ChatML prefix so the quantiser sees realistic
    prompt distributions.
    """
    system = "You are a helpful assistant."
    prefix = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
    return [
        {"input": prefix + "你好！\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "你好！有什么我可以帮助你的吗？"},
        {"input": prefix + "今天天气怎么样？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "今天天气不错，适合出门走走。"},
        {"input": prefix + "What is the capital of France?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "The capital of France is Paris."},
        {"input": prefix + "请用英文翻译：你好\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Hello or Hi in English."},
        {"input": prefix + "简单介绍一下人工智能\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "人工智能是让计算机模拟人类智能的技术，主要包括机器学习、自然语言处理和计算机视觉等领域。"},
        {"input": prefix + "树上有5只鸟，飞走了3只，又来了2只，现在有几只？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "现在树上有4只鸟。5只减去3只是2只，再加上2只是4只。"},
        {"input": prefix + "讲一个简短的笑话\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "为什么程序员不喜欢户外活动？因为阳光里有太多bug。"},
        {"input": prefix + "推荐一道简单的家常菜\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "西红柿炒鸡蛋，简单又美味。准备两个西红柿、三个鸡蛋，先炒鸡蛋盛出，再炒西红柿，最后混合调味即可。"},
        {"input": prefix + "What is 15 multiplied by 7?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "15 multiplied by 7 equals 105."},
        {"input": prefix + "用一句话形容编程\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "编程就是用代码告诉计算机该做什么的过程，既需要逻辑思维也需要创造力。"},
        {"input": prefix + "How do you say 'good morning' in Chinese?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "You say 早上好 (zǎo shang hǎo) for 'good morning' in Chinese."},
        {"input": prefix + "列出三个健康的早餐选择\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "1. 燕麦粥加水果 2. 全麦面包夹鸡蛋 3. 酸奶配坚果和蜂蜜"},
        {"input": prefix + "Write a one-sentence introduction.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Hi there! I'm an AI assistant running on a Rockchip NPU, ready to help with your questions."},
        {"input": prefix + "解释一下光合作用\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "光合作用是植物利用阳光、水和二氧化碳制造养分并释放氧气的过程，是地球上最重要的化学反应之一。"},
        {"input": prefix + "今天适合出门吗？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "如果天气晴朗、气温适中，那就很适合出门活动。建议出门前查看当地的天气预报。"},
        {"input": prefix + "给我讲个故事\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "从前有一只小狐狸，它特别喜欢帮助森林里的动物们。一天，小兔子迷路了，小狐狸带着它穿过树林找到了回家的路。"},
        {"input": prefix + "What's the difference between Python and C++?\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Python is an interpreted, dynamically-typed language great for rapid development. C++ is a compiled, statically-typed language offering finer control over memory and performance."},
        {"input": prefix + "如何提高睡眠质量？\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "保持规律的作息时间，睡前避免使用电子设备，卧室保持安静黑暗，白天适量运动都有助于提高睡眠质量。"},
        {"input": prefix + "Tell me a fun fact.\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible."},
        {"input": prefix + "谢谢你\n<|im_end|>\n<|im_start|>assistant\n",
         "target": "不客气！有什么其他问题随时问我。"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID, e.g. Qwen/Qwen3-0.6B-Instruct")
    parser.add_argument("--target", default="RK3576",
                        choices=["RK3576", "RK3588"],
                        help="Target Rockchip SoC (default: RK3576)")
    parser.add_argument("--quant", default="W4A16",
                        choices=["W4A16", "W8A8", "FP16"],
                        help="Quantisation mode (default: W4A16)")
    parser.add_argument("--max-context", type=int, default=4096,
                        help="Max context length in tokens (default: 4096)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Loading dtype (default: float16 — saves VRAM)")
    parser.add_argument("--calib-dataset", default=None,
                        help="Path to custom calibration JSON "
                        "(default: built-in 20-example set)")
    parser.add_argument("--out", default=None,
                        help="Output filename (default: auto-generated)")
    parser.add_argument("--gpu", default="0",
                        help="CUDA device index (default: 0)")
    args = parser.parse_args()

    # ── Import (may fail if not installed) ───────────────────────────
    try:
        from rkllm.api import RKLLM
    except ImportError:
        print("ERROR: RKLLM-Toolkit not installed.", file=sys.stderr)
        print("Download the SDK from https://console.zbox.filez.com/l/RJJDmB (code: rkllm)", file=sys.stderr)
        print("Then: pip install <path-to-wheel>", file=sys.stderr)
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── Calibration data ─────────────────────────────────────────────
    calib_path = args.calib_dataset
    if not calib_path:
        calib_path = os.path.join(tempfile.gettempdir(), "rkllm_calib.json")
        data = _calibration_data(args.model)
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(data)} calibration examples to {calib_path}")

    # ── NPU cores per platform ───────────────────────────────────────
    npu_cores = 3 if args.target == "RK3588" else 2

    # ── Quant algorithm ──────────────────────────────────────────────
    q_upper = args.quant.upper()
    if q_upper in ("W4A16",):
        quant_algo = "grq"
    else:
        quant_algo = "normal"

    # ── Output filename ──────────────────────────────────────────────
    model_slug = os.path.basename(args.model.rstrip("/")).lower().replace("/", "-")
    out = args.out or f"{model_slug}_{q_upper}_{args.target}.rkllm"

    # ── Load ─────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}  (dtype={args.dtype})")
    llm = RKLLM()
    ret = llm.load_huggingface(
        model=args.model,
        model_lora=None,
        device="cuda",
        dtype=args.dtype,
        custom_config=None,
        load_weight=True,
    )
    if ret != 0:
        print(f"ERROR: load_huggingface returned {ret}", file=sys.stderr)
        return ret

    # ── Build ────────────────────────────────────────────────────────
    print(
        f"Building: quant={q_upper} algo={quant_algo} "
        f"target={args.target} npu_cores={npu_cores} ctx={args.max_context}"
    )
    ret = llm.build(
        do_quantization=(q_upper != "FP16"),
        optimization_level=1,
        quantized_dtype=q_upper if q_upper != "FP16" else "FP16",
        quantized_algorithm=quant_algo if q_upper != "FP16" else "normal",
        target_platform=args.target,
        num_npu_core=npu_cores,
        extra_qparams=None,
        dataset=calib_path,
        hybrid_rate=0,
        max_context=args.max_context,
    )
    if ret != 0:
        print(f"ERROR: build returned {ret}", file=sys.stderr)
        return ret

    # ── Export ───────────────────────────────────────────────────────
    print(f"Exporting → {out}")
    ret = llm.export_rkllm(out)
    if ret != 0:
        print(f"ERROR: export_rkllm returned {ret}", file=sys.stderr)
        return ret

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"Done — {out}  ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
