<!-- docs/usage.md -->
# Usage

## Installation

```bash
conda install -c lescailab nanocdr-x
```

## Commands

### predict_cdrs

```bash
predict_cdrs \
  -i path/to/input.csv \
  -o path/to/output_cdrs.csv
```

- `-i/--input`: CSV with columns identifier,input.
- `-o/--output`: Path to write identifier,predicted_cdr1,predicted_cdr2,predicted_cdr3.

### explain_hidden_states

```bash
explain_hidden_states \
  -i path/to/input.csv \
  -o path/to/hidden_states.csv
```

Outputs a table with one row per non‑pad token, including:

- sequence_id
- position
- token
- pred_class
- hs_dim_0…hs_dim_N

### explain_saliency

```bash
explain_saliency \
  -i path/to/input.csv \
  -o path/to/saliency.csv
```

Outputs a detailed saliency map with columns:

- sequence_id
- time_point
- position
- predicted_class
- saliency