import os
import click
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Locate package installation path
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths inside the Conda environment
DEFAULT_MODEL_NAME = "nanobodies_lstm_sequence_model_trained"
DEFAULT_CHAR_MAPPING_NAME = "char_mapping.csv"
DEFAULT_MODEL_PATH = os.path.join(PACKAGE_DIR, DEFAULT_MODEL_NAME)
DEFAULT_CHAR_MAPPING_PATH = os.path.join(PACKAGE_DIR, DEFAULT_CHAR_MAPPING_NAME)

# Utility functions
def load_character_mapping(char_mapping_path):
    char_mapping_df = pd.read_csv(char_mapping_path)
    return dict(zip(char_mapping_df["chars"], char_mapping_df["integers"]))

def tokenize_sequence(sequence, char_map):
    return [char_map[ch] for ch in sequence]

def trim_padding(input_seq, pred_seq):
    trimmed_input = input_seq.rstrip("Z")
    trimmed_pred = pred_seq[:len(trimmed_input)]
    return trimmed_input, trimmed_pred

def extract_substring_for_digit(haystack, codes, digit):
    start_idx = codes.find(str(digit))
    if start_idx == -1:
        return ""
    end_idx = start_idx
    while end_idx < len(codes) and codes[end_idx] == str(digit):
        end_idx += 1
    return haystack[start_idx:end_idx]

def process_sequences(input_file, output_file, model_path, char_mapping_path):
    """Process input sequences using an LSTM model and extract CDR regions."""
    model = tf.keras.models.load_model(model_path)
    char_to_idx = load_character_mapping(char_mapping_path)
    
    df = pd.read_csv(input_file)
    if "input" not in df.columns:
        raise ValueError("The input CSV file must contain a column named 'input'.")
    if "identifier" not in df.columns:
        raise ValueError("The input CSV file must contain a column named 'identifier'.")
    
    df["input_padded"] = df["input"].apply(lambda seq: seq.ljust(150, "Z"))
    tokenized_sequences = np.array([tokenize_sequence(seq, char_to_idx) for seq in df["input_padded"]])
    x_array = tokenized_sequences.astype("int32")

    full_probs = model.predict(x_array)
    predicted_classes = np.argmax(full_probs, axis=-1)

    trimmed_data = [trim_padding(inp, "".join(map(str, pred))) for inp, pred in zip(df["input_padded"], predicted_classes)]
    df_results = pd.DataFrame(trimmed_data, columns=["input", "predicted_classes"])
    
    # Include the "identifier" column from the input CSV in the output
    df_results.insert(0, "identifier", df["identifier"])

    cdr1, cdr2, cdr3 = "1", "2", "3"
    df_results["predicted_cdr1"] = df_results.apply(lambda row: extract_substring_for_digit(row["input"], row["predicted_classes"], cdr1), axis=1)
    df_results["predicted_cdr2"] = df_results.apply(lambda row: extract_substring_for_digit(row["input"], row["predicted_classes"], cdr2), axis=1)
    df_results["predicted_cdr3"] = df_results.apply(lambda row: extract_substring_for_digit(row["input"], row["predicted_classes"], cdr3), axis=1)

    df_results.to_csv(output_file, index=False)
    print(f"Processing complete. Output saved to {output_file}.")

@click.command()
@click.option('-i', '--input', 'input_file', type=click.Path(exists=True), required=True, help="Path to the input CSV file. Must contain columns 'input' and 'identifier'.")
@click.option('-o', '--output', 'output_file', type=click.Path(), required=True, help="Path to the output CSV file where results will be saved.")
@click.option('--model', type=click.Path(), default=DEFAULT_MODEL_PATH, help="Path to the trained model.")
@click.option('--char_mapping', type=click.Path(), default=DEFAULT_CHAR_MAPPING_PATH, help="Path to the character mapping CSV.")
def main(input_file, output_file, model, char_mapping):
    """Main function that processes sequences based on user-provided arguments."""
    process_sequences(input_file, output_file, model, char_mapping)

if __name__ == '__main__':
    main()