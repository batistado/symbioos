# SymbioOS Phase 2 Cluster Scale-Out (`N = 32..256`) & Theorem Verification Sweeps

## 1. Sweep A — Cluster Concurrency Scale-Out (`N = 128` Highlight)

| Policy (`N = 128`) | TTFT P50 (ms) | TTFT P95 (ms) | TTFT P99 (ms) | HBM Hit (%) | Interactive Hit (%) | HBM Waste (GB·s) | Mean JCT (s) | Goodput (ep/min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lru_cfs` | 59.4 | 437.1 | 931.1 | 60.2% | 79.4% | 1076.5 | 24.75 | 202.0 |
| `continuum_cfs` | 72.5 | 818.4 | 1390.2 | 55.7% | 81.4% | 1015.5 | 24.84 | 201.7 |
| `unison_cfs` | 45.2 | 293.1 | 526.6 | 65.6% | 85.9% | 1037.1 | 24.75 | 202.0 |
| `symbioos_sched` | 42.9 | 296.6 | 436.2 | 69.4% | 82.8% | 691.5 | 24.22 | 216.2 |
| `symbioos_full` | 40.2 | 312.5 | 451.2 | 69.4% | 80.6% | 698.3 | 24.30 | 212.2 |

## 2. Sweep B — HBM Capacity Frontier (`N = 128`, `LRU+CFS` vs `SymbioOS-Full`)

| HBM Ratio | `LRU+CFS` P50 (ms) | `SymbioOS-Full` P50 (ms) | `LRU+CFS` Hit (%) | `SymbioOS-Full` Hit (%) | `LRU+CFS` Goodput | `SymbioOS-Full` Goodput |
|---:|---:|---:|---:|---:|---:|---:|
| `15%` | 3701.9 | **638.1** | 23.6% | **29.9%** | 203.2 | **225.2** |
| `25%` | 506.6 | **131.7** | 39.5% | **49.7%** | 203.6 | **217.9** |
| `37%` | 59.4 | **40.2** | 60.2% | **69.4%** | 202.0 | **212.2** |
| `50%` | 51.4 | **40.8** | 67.3% | **77.6%** | 202.0 | **212.6** |
| `65%` | 40.1 | **38.9** | 77.6% | **82.4%** | 202.0 | **211.7** |

## 3. Sweep D — Theorem 2 JIT Optimal-Stopping Boundary (`N = 128`)

| JIT Margin `delta_jit` (s) | TTFT P50 (ms) | TTFT P95 (ms) | HBM Hit (%) | Premature JIT Pin (GB·s) | Total HBM Waste (GB·s) |
|---:|---:|---:|---:|---:|---:|
| `0.00s` | 42.4 | 288.3 | 68.1% | 0.00 | 682.8 |
| `0.15s` | 40.5 | 287.1 | 67.8% | 0.83 | 692.8 |
| `0.35s` | 40.2 | 312.5 | 69.4% | 6.10 | 698.3 |
| `0.70s` | 37.0 | 273.0 | 75.8% | 17.63 | 707.0 |
| `1.40s` | 36.7 | 219.3 | 87.5% | 55.09 | 707.7 |
| `2.80s` | 36.5 | 158.4 | 94.9% | 144.11 | 697.5 |
