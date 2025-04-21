# explain_saliency CLI

## Description

This script computes **gradient‐based saliency maps** for each non‐padded token in each sequence. For each target time point t, it records how sensitive the model’s predicted class probability is to each input position p, yielding a 2D saliency matrix. The output CSV has one row per (sequence, time_point, position) triple with:

- sequence_id  
- time_point (target step)  
- position (input step)  
- predicted_class at time_point  
- saliency value (gradient magnitude)

## Theoretical background

- **Saliency maps** in deep learning measure the gradient of an output scalar (e.g., class probability) with respect to input features or embeddings.  
- TensorFlow’s **GradientTape** API records operations on tensors; by watching the embedding output and computing `tape.gradient(prob, embedding)`, one obtains how much each embedding dimension influences the probability.  
- Aggregating (e.g., mean absolute value) across embedding dimensions yields a single saliency score per position.  
- Saliency across (t,p) reveals which input residues are most influential for each predicted residue label.

## Usage

```bash
  explain_saliency \
    -i /path/to/input.csv \
    -o /path/to/saliency.csv \
    [--model /path/to/model_dir] \
    [--char_mapping /path/to/char_mapping.csv]
```

## Options

```bash
  -i, --input         path to CSV with 'identifier' and 'input' columns  
  -o, --output        path where saliency CSV will be saved  
  --model             optional override of the model directory  
  --char_mapping      optional override of the char→int mapping CSV  
```