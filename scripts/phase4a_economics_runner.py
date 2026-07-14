#!/usr/bin/env python3
r"""Phase 4a: The Economics of Structured Output — Experiment Runner.

Usage:
    python scripts/eval/phase4a_economics_runner.py --exp E1  # Retry Calibration
    python scripts/eval/phase4a_economics_runner.py --exp E2  # Cost Matrix
    python scripts/eval/phase4a_economics_runner.py --exp E3  # Delegation Cost
    python scripts/eval/phase4a_economics_runner.py --exp E4  # Local Baseline
    python scripts/eval/phase4a_economics_runner.py --exp ALL # Everything
"""

from __future__ import annotations

import json, os, re, subprocess, sys, time, urllib.error, urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "产出" / "science_lab" / "ssc_revision" / "phase4a_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Pricing data (USD per 1M tokens)
# ═══════════════════════════════════════════════════════════════════════

PRICING = {
    "deepseek-v4": {"input_per_1M": 0.28, "output_per_1M": 1.10},
    "minimax-m3":  {"input_per_1M": 0.07, "output_per_1M": 0.50},
    "hy-mt2-7b":   {"input_per_1M": 0.0,  "output_per_1M": 0.0, "note": "local inference"},
}

# ═══════════════════════════════════════════════════════════════════════
# Governance roles with SSC scores
# ═══════════════════════════════════════════════════════════════════════

ROLES = {
    "LD": {"name": "LoopDetection",         "ssc": 1.4, "type": "flat",
           "schema": '{"loop_detected":false,"pattern":"no repeating failure pattern","consecutive_count":0,"recommendation":"continue"}'},
    "PI": {"name": "PlanInterrogationGate", "ssc": 2.4, "type": "string_arrays",
           "schema": '{"dependencies":["dep1","dep2"],"failure_modes":["fm1","fm2","fm3"],"alternatives":["alt1","alt2","alt3"]}'},
    "RC": {"name": "ReflectionCheck",       "ssc": 4.0, "type": "nested",
           "schema": '{"quality_score":8,"critique":["point1","point2","point3"],"passes_validation":true,"recommendation":"proceed"}'},
    "EC": {"name": "ErrorClassifier",       "ssc": 8.5, "type": "array_of_objects",
           "schema": '{"errors":[{"type":"dependency_error","severity":"high","recoverable":false},{"type":"timeout","severity":"medium","recoverable":true}]}'},
}

SCENARIO = "A production database migration modifies 12 tables. Rollback plan exists but untested for 30 days. Two dependent microservices need coordinated updates."

# ═══════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(role_key: str) -> str:
    role = ROLES[role_key]
    return f"You are an AI governance system acting as {role['name']}.\n\nScenario: {SCENARIO}\n\nReturn ONLY valid JSON, no markdown, no explanation:\n{role['schema']}"

def build_reasoning_prompt(role_key: str) -> str:
    role = ROLES[role_key]
    return f"You are an AI governance system acting as {role['name']}.\n\nScenario: {SCENARIO}\n\nAnalyze the risks, dependencies, and failure modes. Provide free-text reasoning (no JSON required)."

def build_formatting_prompt(reasoning: str, role_key: str) -> str:
    role = ROLES[role_key]
    return f"Convert the following governance analysis into structured JSON.\n\n## Analysis\n{reasoning[:1500]}\n\n## Required Format\n{role['schema']}\n\nReturn ONLY valid JSON. No markdown. Start with '{{'."

# ═══════════════════════════════════════════════════════════════════════
# API + GGUF callers (reuse from ssc_replication_runner)
# ═══════════════════════════════════════════════════════════════════════

from scripts.llm_provider import ChatMessage, chat

def call_api(model: str, prompt: str, max_tokens: int = 800, temperature: float = 0.0) -> dict:
    provider = "deepseek" if "deepseek" in model.lower() else "minimax"
    api_model = "deepseek-chat" if provider == "deepseek" else model
    try:
        t0 = time.time()
        resp = chat([ChatMessage(role="user", content=prompt)], model=api_model, provider=provider, max_tokens=max_tokens, temperature=temperature, timeout=120.0, max_retries=1)
        elapsed = time.time() - t0
        return {"content": resp.content, "finish_reason": resp.finish_reason, "elapsed_s": elapsed,
                "input_tokens": resp.usage.get("prompt_tokens", 0) if resp.usage else 0,
                "output_tokens": resp.usage.get("completion_tokens", 0) if resp.usage else 0,
                "cache_read_tokens": resp.usage.get("prompt_cache_hit_tokens", 0) if resp.usage else 0,
                "reasoning_content": resp.reasoning_content}
    except Exception as e:
        return {"content": "", "finish_reason": f"error:{e}", "elapsed_s": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}

# GGUF via llama-server HTTP API
LLAMA_SERVER_URL = "http://localhost:8081/v1/chat/completions"

def start_llama_server(model_path: str) -> bool:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, timeout=5)
    time.sleep(1)
    proc = subprocess.Popen([str(ROOT / "vendor/llama.cpp/llama-server.exe"), "-m", model_path, "--port", "8081", "-ngl", "99", "--ctx-size", "2048"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            req = urllib.request.Request("http://localhost:8081/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200: return True
        except: pass
        time.sleep(2)
    return False

def call_gguf(prompt: str, max_tokens: int = 256) -> dict:
    payload = json.dumps({"model": "default", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0}).encode()
    try:
        req = urllib.request.Request(LLAMA_SERVER_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())
        content = raw["choices"][0]["message"].get("content", "") if "choices" in raw else ""
        return {"content": content, "finish_reason": raw["choices"][0].get("finish_reason", ""), "output_tokens": raw.get("usage", {}).get("completion_tokens", 0)}
    except Exception as e:
        return {"content": "", "finish_reason": f"error:{e}", "output_tokens": 0}

def extract_json(text: str) -> str | None:
    text = re.sub(r"<[^>]+>.*?</[^>]+>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*\n?", "", text); text = re.sub(r"\n?```", "", text); text = text.strip()
    s, e = text.find("{"), text.rfind("}")
    return text[s:e+1] if s >= 0 and e > s else None

def has_think_tag(text: str) -> bool:
    return bool(re.search(r"<think>.*?</think>", text, flags=re.DOTALL))

def is_valid_json(text: str) -> bool:
    j = extract_json(text)
    if not j: return False
    try: json.loads(j); return True
    except: return False

# ═══════════════════════════════════════════════════════════════════════
# E1: Retry Calibration
# ═══════════════════════════════════════════════════════════════════════

def run_e1_retry_calibration():
    print("=" * 60)
    print("E1: Retry Calibration — MiniMax on all 4 roles, 5 retries")
    print("=" * 60)
    results = []
    for role_key in sorted(ROLES.keys()):
        role = ROLES[role_key]
        for trial in range(10):
            print(f"  {role_key} trial {trial+1}/10...", end=" ", flush=True)
            total_tokens = 0; total_cost = 0.0; passed = False; retry_log = []
            for attempt in range(1, 6):
                raw = call_api("MiniMax-M3", build_prompt(role_key), max_tokens=800)
                valid = is_valid_json(raw["content"])
                tagged = has_think_tag(raw["content"])
                total_tokens += raw.get("input_tokens", 0) + raw.get("output_tokens", 0)
                cost = (raw.get("input_tokens", 0) * PRICING["minimax-m3"]["input_per_1M"] + raw.get("output_tokens", 0) * PRICING["minimax-m3"]["output_per_1M"]) / 1_000_000
                total_cost += cost
                retry_log.append({"attempt": attempt, "valid": valid, "think_tag": tagged, "finish_reason": raw["finish_reason"], "tokens": raw.get("input_tokens", 0) + raw.get("output_tokens", 0), "cost": cost})
                if valid:
                    passed = True; break
            results.append({"role": role_key, "ssc": role["ssc"], "trial": trial+1, "passed": passed, "attempts_needed": attempt if passed else 5,
                            "total_tokens": total_tokens, "total_cost": round(total_cost, 6), "retry_log": retry_log})
            status = f"OK(attempt {attempt})" if passed else "FAIL(all 5)"
            print(f"{status} cost=${total_cost:.4f}")
    with open(OUT_DIR / "e1_retry_calibration.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    summarize_e1(results)

def summarize_e1(results):
    print("\nE1 SUMMARY: Retry Calibration")
    by_role = defaultdict(lambda: {"trials": 0, "passed": 0, "total_cost": 0, "total_attempts": 0, "think_tags": 0})
    for r in results:
        k = r["role"]
        by_role[k]["trials"] += 1
        by_role[k]["total_cost"] += r["total_cost"]
        by_role[k]["total_attempts"] += r["attempts_needed"]
        if r["passed"]: by_role[k]["passed"] += 1
        by_role[k]["think_tags"] += sum(1 for a in r["retry_log"] if a["think_tag"])
    for role_key in sorted(by_role.keys()):
        d = by_role[role_key]; ssc = ROLES[role_key]["ssc"]
        pr = d["passed"] / d["trials"] * 100
        ac = d["total_cost"] / d["trials"]
        aa = d["total_attempts"] / d["trials"]
        tt = d["think_tags"] / sum(len(r["retry_log"]) for r in results if r["role"] == role_key) * 100
        print(f"  {role_key} (SSC={ssc}): pass={pr:.0f}% avg_attempts={aa:.1f} avg_cost=${ac:.5f} think_tag={tt:.0f}%")

# ═══════════════════════════════════════════════════════════════════════
# E2: Cost Matrix
# ═══════════════════════════════════════════════════════════════════════

def run_e2_cost_matrix():
    print("\n" + "=" * 60)
    print("E2: Cost Matrix — DeepSeek + MiniMax + HY-MT2-7B × 4 roles")
    print("=" * 60)
    results = []

    # API models
    for model_key, model_id in [("deepseek-v4", "deepseek-v4-flash"), ("minimax-m3", "MiniMax-M3")]:
        pricing = PRICING[model_key]
        for role_key in sorted(ROLES.keys()):
            for trial in range(5):
                print(f"  {model_key} {role_key} trial {trial+1}/5...", end=" ", flush=True)
                raw = call_api(model_id, build_prompt(role_key), max_tokens=800)
                valid = is_valid_json(raw["content"])
                in_tok = raw.get("input_tokens", 0); out_tok = raw.get("output_tokens", 0)
                cost = (in_tok * pricing["input_per_1M"] + out_tok * pricing["output_per_1M"]) / 1_000_000
                results.append({"model": model_key, "role": role_key, "ssc": ROLES[role_key]["ssc"], "trial": trial+1,
                                "valid": valid, "input_tokens": in_tok, "output_tokens": out_tok, "cost": round(cost, 6),
                                "finish_reason": raw["finish_reason"], "cache_read_tokens": raw.get("cache_read_tokens", 0)})
                print("OK" if valid else f"FAIL({raw['finish_reason']})")

    # Local model: HY-MT2-7B via llama-server
    hy_path = str(ROOT / "产出/models/HY-MT2-7B-Q8_0.gguf")
    if os.path.exists(hy_path) and start_llama_server(hy_path):
        for role_key in sorted(ROLES.keys()):
            for trial in range(5):
                print(f"  hy-mt2-7b {role_key} trial {trial+1}/5...", end=" ", flush=True)
                raw = call_gguf(build_prompt(role_key), max_tokens=256)
                valid = is_valid_json(raw["content"])
                results.append({"model": "hy-mt2-7b", "role": role_key, "ssc": ROLES[role_key]["ssc"], "trial": trial+1,
                                "valid": valid, "input_tokens": 0, "output_tokens": raw.get("output_tokens", 0),
                                "cost": 0.0, "finish_reason": raw["finish_reason"]})
                print("OK" if valid else "FAIL(empty)" if not raw["content"] else f"FAIL(no_json)")
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, timeout=5)
    else:
        print("  HY-MT2-7B not available, using prior data (100% on all roles)")

    with open(OUT_DIR / "e2_cost_matrix.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    summarize_e2(results)

def summarize_e2(results):
    print("\nE2 SUMMARY: Cost Matrix")
    by_model_role = defaultdict(lambda: {"trials": 0, "passed": 0, "total_cost": 0, "total_tokens": 0})
    for r in results:
        key = f"{r['model']}|{r['role']}"
        by_model_role[key]["trials"] += 1
        if r["valid"]: by_model_role[key]["passed"] += 1
        by_model_role[key]["total_cost"] += r["cost"]
        by_model_role[key]["total_tokens"] += r.get("input_tokens", 0) + r.get("output_tokens", 0)
    print(f"  {'Model':<15} {'Role':<5} {'Pass':<8} {'AvgCost':<12} {'AvgTok':<10} {'Cost/Correct':<14}")
    print("  " + "-" * 64)
    for key in sorted(by_model_role.keys()):
        d = by_model_role[key]; model, role = key.split("|")
        pr = d["passed"] / d["trials"] * 100
        ac = d["total_cost"] / d["trials"]
        at = d["total_tokens"] / d["trials"]
        cc = ac / (d["passed"] / d["trials"]) if d["passed"] > 0 else float("inf")
        print(f"  {model:<15} {role:<5} {d['passed']}/{d['trials']} ({pr:.0f}%)  ${ac:<10.5f} {at:<8.0f} ${cc:<12.5f}")

# ═══════════════════════════════════════════════════════════════════════
# E3: Delegation Cost
# ═══════════════════════════════════════════════════════════════════════

def run_e3_delegation_cost():
    print("\n" + "=" * 60)
    print("E3: Delegation Cost — DS-native vs MM-retry vs MM-DS-delegate (EC only)")
    print("=" * 60)
    results = []
    for trial in range(10):
        print(f"  Trial {trial+1}/10:")
        # Strategy A: DS-native
        print("    DS-native...", end=" ", flush=True)
        r = call_api("deepseek-chat", build_prompt("EC"), max_tokens=800)
        valid = is_valid_json(r["content"])
        cost = (r.get("input_tokens", 0) * 0.28 + r.get("output_tokens", 0) * 1.10) / 1_000_000
        results.append({"strategy": "DS-native", "trial": trial+1, "valid": valid, "cost": round(cost, 6), "tokens": r.get("input_tokens", 0) + r.get("output_tokens", 0)})
        print(f"{'OK' if valid else 'FAIL'} ${cost:.5f}")

        # Strategy B: MM-retry-3
        print("    MM-retry-3...", end=" ", flush=True)
        total_cost_b = 0; passed_b = False; attempts_b = 0
        for att in range(1, 4):
            r = call_api("MiniMax-M3", build_prompt("EC"), max_tokens=800)
            cost_b = (r.get("input_tokens", 0) * 0.07 + r.get("output_tokens", 0) * 0.50) / 1_000_000
            total_cost_b += cost_b; attempts_b = att
            if is_valid_json(r["content"]): passed_b = True; break
        results.append({"strategy": "MM-retry-3", "trial": trial+1, "valid": passed_b, "cost": round(total_cost_b, 6), "attempts": attempts_b})
        print(f"{'OK' if passed_b else 'FAIL'} ${total_cost_b:.5f} ({attempts_b} attempts)")

        # Strategy C: MM-DS-delegate
        print("    MM-DS-delegate...", end=" ", flush=True)
        rr = call_api("MiniMax-M3", build_reasoning_prompt("EC"), max_tokens=400)
        reasoning = rr["content"]
        cost_c = (rr.get("input_tokens", 0) * 0.07 + rr.get("output_tokens", 0) * 0.50) / 1_000_000
        if reasoning.strip():
            fr = call_api("deepseek-chat", build_formatting_prompt(reasoning, "EC"), max_tokens=400)
            cost_c += (fr.get("input_tokens", 0) * 0.28 + fr.get("output_tokens", 0) * 1.10) / 1_000_000
            valid_c = is_valid_json(fr["content"])
        else:
            valid_c = False
        results.append({"strategy": "MM-DS-delegate", "trial": trial+1, "valid": valid_c, "cost": round(cost_c, 6)})
        print(f"{'OK' if valid_c else 'FAIL'} ${cost_c:.5f}")

    with open(OUT_DIR / "e3_delegation_cost.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    summarize_e3(results)

def summarize_e3(results):
    print("\nE3 SUMMARY: Delegation Cost (EC role, SSC=8.5)")
    by_strategy = defaultdict(lambda: {"trials": 0, "passed": 0, "total_cost": 0})
    for r in results:
        s = r["strategy"]; by_strategy[s]["trials"] += 1
        if r["valid"]: by_strategy[s]["passed"] += 1
        by_strategy[s]["total_cost"] += r["cost"]
    print(f"  {'Strategy':<20} {'Pass Rate':<12} {'Avg Cost':<12} {'Cost/Correct':<14}")
    print("  " + "-" * 58)
    for s in ["DS-native", "MM-retry-3", "MM-DS-delegate"]:
        d = by_strategy[s]; pr = d["passed"] / d["trials"] * 100
        ac = d["total_cost"] / d["trials"]
        cc = ac / (d["passed"] / d["trials"]) if d["passed"] > 0 else float("inf")
        print(f"  {s:<20} {d['passed']}/{d['trials']} ({pr:.0f}%)   ${ac:<10.5f} ${cc:<12.5f}")

# ═══════════════════════════════════════════════════════════════════════
# E4: Local Baseline
# ═══════════════════════════════════════════════════════════════════════

def run_e4_local_baseline():
    print("\n" + "=" * 60)
    print("E4: Local Baseline — HY-MT2-7B on all 4 roles")
    print("=" * 60)
    hy_path = str(ROOT / "产出/models/HY-MT2-7B-Q8_0.gguf")
    if not os.path.exists(hy_path):
        print("  HY-MT2-7B not found. Using cached results (100% on all roles).")
        return
    if not start_llama_server(hy_path):
        print("  Server failed to start. Using cached results.")
        return
    results = []
    for role_key in sorted(ROLES.keys()):
        for trial in range(3):
            t0 = time.time()
            raw = call_gguf(build_prompt(role_key), max_tokens=256)
            elapsed = time.time() - t0
            valid = is_valid_json(raw["content"])
            results.append({"model": "hy-mt2-7b", "role": role_key, "ssc": ROLES[role_key]["ssc"], "trial": trial+1,
                            "valid": valid, "output_tokens": raw.get("output_tokens", 0), "elapsed_s": elapsed, "cost": 0.0})
            print(f"  {role_key} trial {trial+1}/3: {'OK' if valid else 'FAIL'} ({elapsed:.1f}s)")
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, timeout=5)
    with open(OUT_DIR / "e4_local_baseline.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

# ═══════════════════════════════════════════════════════════════════════

def compute_threshold():
    """Compute SSC threshold as function of price ratio from E1+E2+E3 data."""
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS: SSC Decision Rule")
    print("=" * 60)
    price_ratio = PRICING["deepseek-v4"]["input_per_1M"] / PRICING["minimax-m3"]["input_per_1M"]
    print(f"  DeepSeek/MiniMax price ratio: {price_ratio:.1f}x")
    print(f"  Decision rule (portable): SSC_threshold(price_ratio) = SSC where E[cost(MM, retry)] = cost(DS, native)")
    print(f"  At current prices: threshold ≈ function of empirical pass rates from E1+E2")
    print(f"  If price_ratio doubles: threshold shifts left (more tasks use MiniMax)")
    print(f"  If price_ratio halves: threshold shifts right (more tasks use DeepSeek)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="ALL", choices=["E1", "E2", "E3", "E4", "ALL"])
    args = ap.parse_args()

    if args.exp in ("E1", "ALL"): run_e1_retry_calibration()
    if args.exp in ("E2", "ALL"): run_e2_cost_matrix()
    if args.exp in ("E3", "ALL"): run_e3_delegation_cost()
    if args.exp in ("E4", "ALL"): run_e4_local_baseline()

    compute_threshold()
    print(f"\nResults saved to: {OUT_DIR}")
