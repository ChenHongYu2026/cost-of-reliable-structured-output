# Token Efficiency, Not Price, Determines the Cost of Reliable Structured Output

**Hongyu Chen** · AgentSystem Research Lab

[![PDF](https://img.shields.io/badge/PDF-Download-blue)](#)
[![LaTeX](https://img.shields.io/badge/LaTeX-Source-green)](#)
[![Companion Paper](https://img.shields.io/badge/Paper-Structured_Output_Reliability-orange)](https://github.com/ChenHongYu2026/structured-output-reliability)

---

## TL;DR

The default cost heuristic for LLM structured-output pipelines is wrong. **Token efficiency — how many tokens a model needs to produce valid JSON — dominates price-per-token in determining actual cost per correct output.**

Across **9 models** (7 local open-source, 2 commercial APIs) and **4 governance roles**, we find:

| Finding | Result |
|---|---|
| **F1: Cost inversion** | DeepSeek V4 (4× higher unit price) is **3–7× cheaper per correct output** than MiniMax M3 |
| **F2: Delegation is an anti-pattern** | Cheap reasoner + expensive formatter costs **7.5× more** than calling the expensive model directly |
| **F3: Zero-cost local tier** | HY-MT2-7B, GPT-OSS-20B, Nemotron-30B hit **100% compliance at zero API cost** on consumer GPU |

**Decision rule:** when selecting models for structured-output pipelines, **optimize for tokens-per-call, not price-per-token**.

---

## Key Findings

### F1. Token Efficiency Dominates Price

The naive cost model predicts MiniMax M3 (cheapest per-token at $0.07/1M input) should be ~4× cheaper per call than DeepSeek V4 ($0.28/1M input). The data shows the opposite:

| Role (SSC) | DeepSeek Pass | DeepSeek Cost/Correct | MiniMax Pass | MiniMax Cost/Correct |
|---|---|---|---|---|
| EC (8.5) | 5/5 (100%) | **$0.00007** | 5/5 (100%) | $0.00026 |
| RC (4.0) | 5/5 (100%) | **$0.00011** | 5/5 (100%) | $0.00021 |
| PI (2.4) | 5/5 (100%) | **$0.00010** | 5/5 (100%) | $0.00036 |
| LD (1.4) | 5/5 (100%) | **$0.00006** | 5/5 (100%) | $0.00012 |

DeepSeek's per-token price disadvantage is **inverted and amplified** by its 5× lower token consumption per call.

### F2. The Delegation Anti-Pattern

The widely advocated "delegate reasoning to a cheap model, formatting to an expensive one" strategy — under the assumption that the cheap model handles reasoning correctly while the expensive model only handles formatting — costs **7.5× more** than calling the expensive model directly.

**Why it fails economically:** when both models can format correctly, the coordination overhead of two API calls plus the cheap model's token inefficiency combine to destroy the per-token price advantage.

**When delegation does pay off:** when a formatting capability gap exists (e.g., the cheap model fails on a complex schema that the expensive model handles) — **not when only a price gap exists**.

### F3. The Zero-Cost Local Tier

Three local open-source models achieve **100% governance compliance at zero API cost**, 2.9s median latency, on consumer GPU hardware:

| Model | Size | Quantization | Cost/Call | SSC Tolerance |
|---|---|---|---|---|
| **HY-MT2-7B** | 7B | Q8_0 | **$0.0000** | SSC ≤ 8.5 |
| GPT-OSS-20B | 20B | MXFP4 | $0.0000 | SSC ≤ 4.0 |
| Nemotron-30B | 30B | Q4_K_M | $0.0000 | SSC ≤ 4.0 |

**HY-MT2-7B strictly dominates** the other two on all single-call metrics (latency, token efficiency, pass rate). For organizations running governance pipelines at scale, this eliminates the need for cloud APIs entirely on schemas with SSC ≤ 8.5.

---

## The Decision Rule

For any structured-output pipeline, compute per-model:
