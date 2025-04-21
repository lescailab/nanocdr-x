# predict_cdrs CLI

## Description:
This script takes an input CSV with columns 'identifier' and 'input' (amino-acid sequences), pads each sequence to a fixed length (150) using the 'Z' character, tokenizes each character via a pre‐built mapping, and runs the sequences through a trained TensorFlow LSTM model. The model outputs per‐position class predictions, which are then trimmed of padding and used to extract the three complementarity‐determining region (CDR) segments (CDR1, CDR2, CDR3) via simple substring extraction functions. Results are saved to an output CSV.

## Theoretical background:
- CDRs are the hypervariable loops in antibody/nanobody sequences that determine binding specificity.  
- The model architecture: an Embedding layer → two bidirectional LSTM layers → TimeDistributed Dense for per‐position classification.  
- Padding to a fixed length ensures uniform input shape; predictions beyond the real sequence (padded) are discarded via trimming.  
- Output classes correspond to integer labels that denote CDR membership or non‐CDR.  

## Usage:

```bash
  predict_cdrs \
    -i /path/to/input.csv \
    -o /path/to/output_cdrs.csv \
    [--model /path/to/model_dir] \
    [--char_mapping /path/to/char_mapping.csv]
```

## Options:

```bash
  -i, --input         path to CSV with 'identifier' and 'input' columns  
  -o, --output        path where output CSV with predicted CDR columns will be saved  
  --model             optional override of the model directory (default from package)  
  --char_mapping      optional override of the char→int mapping CSV (default from package)
```