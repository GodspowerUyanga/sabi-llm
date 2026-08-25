# Sabi v1 — ADTC Compliance Report

**Status: 6/7 checks passed**

## Constraint checklist

| Check | Result | Detail |
|---|:--:|---|
| Runs 100% offline (no network at inference) | ✅ | On-device llama.cpp; no external calls during inference. |
| Peak RAM under 7 GB budget | ✅ | Peak 1.944 GB of 7 GB (13.1 GB free). |
| GGUF model present (judge-loadable) | ✅ | sabi-1.gguf · 1065.6 MB |
| Core temperature under 85 °C | ❌ | 101.0 °C |
| CPU-only, no discrete GPU required | ✅ | Intel(R) Core(TM) Ultra 7 165H |
| Deterministic accuracy on data (no hallucinated maths) | ✅ | Totals, counts, debtors, pivots computed in code, not by the model. |
| English (primary) + Nigerian Pidgin | ✅ | Focused languages — accuracy over breadth, per ADTC FAQ. |

## Model

- Name: **Sabi-1**
- Format: GGUF · quantization q4_k_m
- Weight on disk: **1065.6 MB**
- Context window: 8192 tokens

## Telemetry (this machine)

- Peak RAM: **1.944 GB** of 7 GB budget
- Efficiency score S_eff ≈ **72.2** / 100
- Generation speed: — run `python -m sabi bench` tokens/sec
- Core temperature: 101.0 °C

## Hardware

- CPU: Intel(R) Core(TM) Ultra 7 165H
- OS: Ubuntu 22.04.5 LTS
- Arch: x86_64 · Python 3.10.12

## Scoring model

`S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`

_Speed and Efficiency are measured automatically on the target laptop; run `python -m sabi bench` to populate the live speed figure._