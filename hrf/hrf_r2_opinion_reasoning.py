# hrf/hrf_r2_opinion_reasoning.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict

from openai import OpenAI

from .prompts import get_prompts


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str = "deepseek-chat"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    system_prompt: str = "You are a helpful assistant. Follow the instruction and keep output concise."

    @staticmethod
    def from_env() -> "LLMConfig":
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "").strip()
        model = os.getenv("LLM_MODEL", "deepseek-chat").strip()
        temperature = float(os.getenv("TEMPERATURE", "0.0"))
        top_p = float(os.getenv("TOP_P", "1.0"))
        max_tokens = int(os.getenv("MAX_TOKENS", "256"))
        system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful assistant. Follow the instruction and keep output concise."
        )
        if not api_key:
            raise ValueError("Missing LLM_API_KEY in environment.")
        if not base_url:
            raise ValueError("Missing LLM_BASE_URL in environment.")
        return LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )


def build_client(cfg: LLMConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


def llm(client: OpenAI, cfg: LLMConfig, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def run_hrf(
    client: OpenAI,
    cfg: LLMConfig,
    mode: str,
    step: str,
    S: str,
    I_CLIP: str,
    A: Optional[str] = None,
    R1: Optional[str] = None,
    R2: Optional[str] = None,
) -> Dict[str, str]:
    prompts = get_prompts(mode)

    if mode == "aspect" and not A:
        raise ValueError("In aspect mode, you must provide A (aspect candidates).")

    out: Dict[str, str] = {}

    if step in ("step1", "all"):
        p1 = prompts["p1"].format(S=S, I_CLIP=I_CLIP, A=A or "")
        out["R1"] = llm(client, cfg, p1)

    if step in ("step2", "all"):
        r1 = out.get("R1") if step == "all" else (R1 or "")
        if not r1:
            raise ValueError("step2 requires R1. Run step1 first or pass R1.")
        p2 = prompts["p2"].format(S=S, I_CLIP=I_CLIP, A=A or "", R1=r1)
        out["R2"] = llm(client, cfg, p2)

    if step in ("step3", "all"):
        r2 = out.get("R2") if step == "all" else (R2 or "")
        if not r2:
            raise ValueError("step3 requires R2. Run step2 first or pass R2.")
        p3 = prompts["p3"].format(S=S, I_CLIP=I_CLIP, A=A or "", R2=r2)
        r3 = llm(client, cfg, p3).lower().strip()
        # normalize
        for lab in ("positive", "neutral", "negative"):
            if lab in r3:
                r3 = lab
                break
        out["R3"] = r3

    return out


def main():
    ap = argparse.ArgumentParser(description="HRF reasoning runner (R1/R2/R3) for HRE-FMSA reproduction.")
    ap.add_argument("--mode", choices=["sentence", "aspect"], default="sentence")
    ap.add_argument("--step", choices=["step1", "step2", "step3", "all"], default="step2")
    ap.add_argument("--S", required=True, help="Input sentence text")
    ap.add_argument("--I_clip", required=True, help="Image caption/description (I_CLIP)")
    ap.add_argument("--A", default=None, help="Aspect candidates (only for aspect mode)")
    ap.add_argument("--R1", default=None, help="Provide R1 if running step2 only")
    ap.add_argument("--R2", default=None, help="Provide R2 if running step3 only")
    args = ap.parse_args()

    cfg = LLMConfig.from_env()
    client = build_client(cfg)
    out = run_hrf(
        client=client,
        cfg=cfg,
        mode=args.mode,
        step=args.step,
        S=args.S,
        I_CLIP=args.I_clip,
        A=args.A,
        R1=args.R1,
        R2=args.R2,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
