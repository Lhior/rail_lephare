# Backend benchmarks

Accuracy-gated wall-time comparison plus GPU throughput with utilization sampling.

## Run

```bash
cd rail_lephare  # lephare must include lephare.accel
export OMP_NUM_THREADS=20
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_X64=1
export CUDA_VISIBLE_DEVICES=0
pip install -e '.[jax,cuda]'

# Full suite (parity + short probe + 1M large throughput)
python benchmarks/bench_backends.py

# Large in-memory clones only (1M galaxies, 100k RAIL chunks)
python benchmarks/bench_backends.py --skip-parity --skip-probe \
  --large-n 1000000 --chunk-size 100000 \
  --backends jax_hybrid cuda_hybrid
```

## What it measures

### Parity fixture (~1.5k objects)

| Backend | FP64 | FP32 |
|---------|------|------|
| `lephare` | inform + estimate (accuracy gated) | — |
| `jax_*`, `cuda_*` | inform + estimate (accuracy gated) | timing only |

FP64 rows must `np.isclose` the golden PDF sum `99.66482986408954` or the script exits.

### Throughput probe (one GPU-sized tile)

- Chooses the largest safe object chunk from free VRAM (`suggest_chi2_chunk_size`).
- Times **estimate-only** for each backend/dtype.
- Samples `nvidia-smi` util/memory.

### Large throughput (default: 1M galaxies, 100k chunks)

- Tiles the test HDF5 **in memory** to `--large-n` rows (photometry clones).
- Manually slices into `--chunk-size` blocks (default 100 000). RAIL’s own
  `chunk_size` is ignored for in-memory tables, so the bench does the slicing.
- GPU backends further sub-chunk each slice to fit VRAM.
- Skips `lephare` (would take hours at 1M).
- Default dtype is FP64 only (`--large-dtypes float64 float32` to add FP32).

**Engagement heuristic (probe):** peak util ≳ 50% on max-batch JAX/CUDA estimate indicates the GPU is meaningfully occupied.
