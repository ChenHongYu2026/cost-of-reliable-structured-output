#!/usr/bin/env python3
r"""Phase 4a GGUF Full Profile — 7 local models with rich prompts, 800 tokens.

Iron Law 1: Rich contextual prompts (not minimal)
Iron Law 2: max_tokens=800 for all calls (eliminate truncation)
Iron Law 3: Record prompt_tokens, generated_tokens, tokens_per_second

Usage:
    python scripts/eval/phase4a_gguf_full.py
"""

from __future__ import annotations

import json, os, re, subprocess, sys, time, urllib.error, urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LLAMA_SERVER = ROOT / "vendor" / "llama.cpp" / "llama-server.exe"
MODELS_DIR = ROOT / "产出" / "models"
OUT_DIR = ROOT / "产出" / "science_lab" / "ssc_revision" / "phase4a_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Rich Contextual Prompts — same format as API experiments (EXP-08-CLEAN)
# ═══════════════════════════════════════════════════════════════════════

SCENARIO = (
    "A software deployment pipeline has been triggered for a production database migration. "
    "The migration script modifies 12 tables including the `users` and `transactions` tables. "
    "A rollback plan exists but has not been tested in the last 30 days. "
    "The deployment window is 2:00-4:00 AM UTC with low user traffic. "
    "Two dependent microservices (payment-gateway, notification-service) need coordinated updates."
)

def build_rich_prompt(role_key: str) -> str:
    role = ROLES[role_key]
    return (
        f"You are an AI governance system performing a structured safety check.\n\n"
        f"## Task Context\n"
        f"You are acting as the {role['name']} layer in a multi-layer agent harness. "
        f"Your job is to analyze the following scenario and produce a structured JSON verdict.\n\n"
        f"## Scenario\n{SCENARIO}\n\n"
        f"## Output Instructions\n"
        f"Return ONLY valid JSON. No markdown, no code fences, no explanatory text.\n"
        f"The JSON must conform exactly to this schema:\n\n{role['schema']}\n\n"
        f"Your response must start with '{{' and end with '}}'.\n"
        f"Do NOT include think tags or any text outside the JSON object."
    )

ROLES = {
    "LD": {"name": "LoopDetection",         "ssc": 1.4,
           "schema": '{"loop_detected":false,"pattern":"no repeating failure pattern observed","consecutive_count":0,"recommendation":"continue"}'},
    "PI": {"name": "PlanInterrogationGate", "ssc": 2.4,
           "schema": '{"dependencies":["dep1","dep2"],"failure_modes":["fm1","fm2","fm3"],"alternatives":["alt1","alt2","alt3"]}'},
    "RC": {"name": "ReflectionCheck",       "ssc": 4.0,
           "schema": '{"quality_score":8,"critique":["point1","point2","point3"],"passes_validation":true,"recommendation":"proceed"}'},
    "EC": {"name": "ErrorClassifier",       "ssc": 8.5,
           "schema": '{"errors":[{"type":"dependency_error","severity":"high","recoverable":false},{"type":"timeout","severity":"medium","recoverable":true}]}'},
}

# ═══════════════════════════════════════════════════════════════════════
# llama-server management
# ═══════════════════════════════════════════════════════════════════════

def kill_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, timeout=5)

def start_server(model_path: str, port: int = 8081, ctx_size: int = 2048) -> bool:
    kill_server()
    time.sleep(1)
    proc = subprocess.Popen(
        [str(LLAMA_SERVER), "-m", model_path, "--port", str(port),
         "-ngl", "99", "--ctx-size", str(ctx_size), "--metrics"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200: return True
        except: pass
        time.sleep(2)
    return False

def call_gguf(prompt: str, max_tokens: int = 800, port: int = 8081) -> dict:
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    try:
        t0 = time.time()
        req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                       data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = json.loads(resp.read())
        elapsed = time.time() - t0
        choice = raw["choices"][0] if "choices" in raw else {}
        content = choice.get("message", {}).get("content", "") or ""
        finish = choice.get("finish_reason", "")
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", len(content.split()) if content else 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0
        return {"content": content, "finish_reason": finish, "elapsed_s": elapsed,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "tokens_per_sec": tps}
    except Exception as e:
        return {"content": "", "finish_reason": f"error:{e}", "elapsed_s": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "tokens_per_sec": 0}

# ═══════════════════════════════════════════════════════════════════════
# JSON validation
# ═══════════════════════════════════════════════════════════════════════

def is_valid_json(text: str) -> bool:
    text = re.sub(r"<[^>]+>.*?</[^>]+>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    text = text.strip()
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s: return False
    try: json.loads(text[s:e+1]); return True
    except: return False

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    gguf_files = sorted(MODELS_DIR.glob("*.gguf"), key=lambda f: f.stat().st_size)
    if not gguf_files:
        print(f"ERROR: No .gguf files in {MODELS_DIR}")
        return

    print(f"Phase 4a GGUF Full Profile — Rich Prompts, max_tokens=800")
    print(f"Models: {len(gguf_files)} | Roles: {len(ROLES)} | Trials: 3 | Total: {len(gguf_files)*len(ROLES)*3} inferences")
    print(f"Iron Law 1: Rich prompts  |  Iron Law 2: max_tokens=800  |  Iron Law 3: Record token efficiency")
    print()

    results = []

    for gguf_path in gguf_files:
        model_name = gguf_path.stem
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"{'='*60}")
        print(f"MODEL: {model_name} ({size_gb:.1f} GB)")
        print(f"{'='*60}")

        if not start_server(str(gguf_path), ctx_size=2048):
            print("  Server failed to start. Skipping.")
            for role_key in ROLES:
                for t in range(3):
                    results.append({"model": model_name, "role": role_key, "ssc": ROLES[role_key]["ssc"],
                                    "trial": t+1, "valid": False, "error": "server_failed"})
            continue

        for role_key in sorted(ROLES.keys()):
            role = ROLES[role_key]
            prompt = build_rich_prompt(role_key)
            prompt_len = len(prompt)
            for trial in range(3):
                print(f"  {role_key} (SSC={role['ssc']}) trial {trial+1}/3...", end=" ", flush=True)
                raw = call_gguf(prompt, max_tokens=800)
                valid = is_valid_json(raw["content"])
                preview = raw["content"][:100] if raw["content"] else ""

                results.append({
                    "model": model_name, "role": role_key, "ssc": role["ssc"],
                    "trial": trial + 1, "valid": valid,
                    "prompt_tokens": raw["prompt_tokens"],
                    "completion_tokens": raw["completion_tokens"],
                    "total_tokens": raw["prompt_tokens"] + raw.get("completion_tokens", 0),
                    "tokens_per_sec": round(raw["tokens_per_sec"], 1),
                    "elapsed_s": round(raw["elapsed_s"], 1),
                    "finish_reason": raw["finish_reason"],
                    "prompt_length_chars": prompt_len,
                    "content_preview": preview,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                status = "OK" if valid else f"FAIL({raw['finish_reason'][:20]})"
                tok_info = f"pt={raw['prompt_tokens']} ct={raw['completion_tokens']} {raw['tokens_per_sec']:.0f}t/s"
                print(f"{status} | {raw['elapsed_s']:.1f}s | {tok_info}")

        kill_server()
        time.sleep(1)

        # Save incrementally
        with open(OUT_DIR / "phase4a_gguf_full_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ═══════════════════════════ Summary ═══════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 4A GGUF FULL PROFILE — SUMMARY")
    print(f"{'='*80}")

    by_model_role = defaultdict(lambda: {"trials": 0, "passed": 0, "total_prompt": 0, "total_completion": 0, "total_time": 0})
    for r in results:
        key = f"{r['model']}|{r['role']}"
        by_model_role[key]["trials"] += 1
        if r["valid"]: by_model_role[key]["passed"] += 1
        by_model_role[key]["total_prompt"] += r.get("prompt_tokens", 0)
        by_model_role[key]["total_completion"] += r.get("completion_tokens", 0)
        by_model_role[key]["total_time"] += r.get("elapsed_s", 0)

    # Print per-model summary
    models = sorted(set(r["model"] for r in results))
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        total = len(model_results)
        passed = sum(1 for r in model_results if r["valid"])
        avg_prompt = sum(r.get("prompt_tokens", 0) for r in model_results) / total
        avg_completion = sum(r.get("completion_tokens", 0) for r in model_results) / total
        avg_time = sum(r.get("elapsed_s", 0) for r in model_results) / total
        print(f"\n  {model[:50]}: {passed}/{total} ({passed/total*100:.0f}%) | avg pt={avg_prompt:.0f} ct={avg_completion:.0f} | {avg_time:.1f}s avg")
        # Per-role breakdown
        for role_key in sorted(ROLES.keys()):
            rr = [r for r in model_results if r["role"] == role_key]
            p = sum(1 for r in rr if r["valid"])
            ac = sum(r.get("completion_tokens", 0) for r in rr) / len(rr) if rr else 0
            print(f"    {role_key} (SSC={ROLES[role_key]['ssc']}): {p}/{len(rr)} avg_ct={ac:.0f}")

    print(f"\nResults: {OUT_DIR / 'phase4a_gguf_full_results.json'}")

if __name__ == "__main__":
    main()
