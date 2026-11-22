# SymbioOS Phase 1 Real-DevTool (`pytest` + `clang++` + `git`) & Multi-Architecture Results (Apple M5 Pro)

## 1. Multi-Architecture 5-Policy Benchmark (`os.killpg` POSIX Process Groups)

| Model (`fp16` MPS) | Policy | TTFT P50 (ms) | TTFT P95 (ms) | HBM Hit (%) | Interactive Hit (%) | HBM Waste (MB·s) | Mean JCT (s) | Goodput (ep/min) | SLO Goodput (`<=600ms`/min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-3B-Instruct` | `lru_cfs` | 1209.6 | 1973.2 | 43.8% | 64.3% | 2199.1 | 8.70 | 52.26 | 26.1 |
| `Qwen2.5-3B-Instruct` | `continuum_cfs` | 1265.7 | 2077.1 | 40.6% | 59.5% | 1649.3 | 8.95 | 50.79 | 22.2 |
| `Qwen2.5-3B-Instruct` | `unison_cfs` | 1099.4 | 1800.3 | 54.7% | 73.8% | 1735.5 | 9.20 | 50.64 | 25.3 |
| `Qwen2.5-3B-Instruct` | `symbioos_sched` | 357.1 | 913.5 | 70.3% | 95.2% | 733.8 | 8.10 | 54.36 | 83.2 |
| `Qwen2.5-3B-Instruct` | `symbioos_full` | 262.6 | 868.6 | 68.8% | 95.2% | 695.5 | 7.97 | 53.58 | 90.4 |
| `Qwen2.5-7B-Instruct` | `lru_cfs` | 4355.2 | 7566.7 | 26.6% | 35.7% | 2293.7 | 14.81 | 32.13 | 3.0 |
| `Qwen2.5-7B-Instruct` | `continuum_cfs` | 4064.3 | 6201.1 | 23.4% | 33.3% | 2094.6 | 14.36 | 33.72 | 3.2 |
| `Qwen2.5-7B-Instruct` | `unison_cfs` | 3918.0 | 5872.2 | 29.7% | 42.9% | 1765.3 | 14.43 | 34.19 | 2.1 |
| `Qwen2.5-7B-Instruct` | `symbioos_sched` | 1765.1 | 3552.4 | 54.7% | 81.0% | 1327.0 | 13.26 | 37.38 | 4.7 |
| `Qwen2.5-7B-Instruct` | `symbioos_full` | 2210.5 | 3651.7 | 50.0% | 76.2% | 1323.4 | 12.67 | 38.55 | 8.4 |
| `OLMoE-1B-7B-0924-Instruct` | `lru_cfs` | 2216.3 | 3004.1 | 32.8% | 47.6% | 7671.7 | 10.33 | 45.95 | 10.1 |
| `OLMoE-1B-7B-0924-Instruct` | `continuum_cfs` | 2021.5 | 2977.2 | 37.5% | 54.8% | 5641.4 | 10.24 | 46.72 | 10.2 |
| `OLMoE-1B-7B-0924-Instruct` | `unison_cfs` | 2085.7 | 3124.0 | 39.1% | 57.1% | 5053.2 | 10.17 | 47.42 | 11.9 |
| `OLMoE-1B-7B-0924-Instruct` | `symbioos_sched` | 911.0 | 1494.6 | 67.2% | 90.5% | 2577.6 | 8.04 | 55.13 | 29.3 |
| `OLMoE-1B-7B-0924-Instruct` | `symbioos_full` | 580.9 | 1651.6 | 64.1% | 88.1% | 2827.3 | 8.01 | 55.88 | 57.6 |

## 2. Streaming `stdout` Micro-Prefill Chunk-Size Ablation (`Qwen2.5-3B-Instruct`)

| Chunk Setting | TTFT P50 (ms) | TTFT P95 (ms) | Mean EOF Prefill+Decode (ms) | HBM Hit (%) | Mean JCT (s) | Micro-Chunks Prefilled |
|---|---:|---:|---:|---:|---:|---:|
| `inf (No Stream / SymbioOS-Sched)` | 357.1 | 913.5 | 249.6 | 70.3% | 8.10 | 0 |
| `128 tokens` | 233.2 | 883.3 | 150.4 | 67.2% | 7.88 | 119 |
| `256 tokens (Default)` | 262.6 | 868.6 | 160.2 | 68.8% | 7.97 | 51 |
| `512 tokens` | 391.0 | 1092.8 | 185.5 | 65.6% | 8.15 | 23 |
