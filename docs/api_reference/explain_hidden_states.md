# explain_hidden_states CLI

## Description

This script reads the same input CSV, pads and tokenizes sequences, then extracts the hidden‐state activations from the second bidirectional LSTM layer for every non‐padded token. It produces a long‐form CSV where each row corresponds to one token in one sequence, annotated with:

- sequence_id  
- position (time step)  
- token  
- predicted class  
- hs_dim_0 … hs_dim_{H-1} (one column per hidden dimension)

## Theoretical background

- In RNNs (including LSTM), **hidden states** are the internal memory at each time step, capturing context from past inputs.  
- A bidirectional LSTM maintains two hidden vectors per time step (forward + backward). Here we extract the concatenated output of the last LSTM layer.  
- To retrieve hidden states, we re‐construct a sub‐model in Keras: define a new `Model(inputs=original_input, outputs=second_LSTM_output)` so calling `predict()` yields hidden activations instead of final classifications.  
- These hidden vectors can be used for down‐stream interpretability, clustering, visualization, or as features in other predictive models.

## Usage

```bash
  explain_hidden_states \
    -i /path/to/input.csv \
    -o /path/to/hidden_states.csv \
    [--model /path/to/model_dir] \
    [--char_mapping /path/to/char_mapping.csv]
```

## Options

```bash
  -i, --input         path to CSV with 'identifier' and 'input' columns  
  -o, --output        path where hidden‐states CSV will be saved  
  --model             optional override of the model directory  
  --char_mapping      optional override of the char→int mapping CSV  
```