import os
import click
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
import warnings
from pandas.errors import PerformanceWarning

# turn off the "highly fragmented" warning
warnings.simplefilter("ignore", category=PerformanceWarning)

# Locate package installation path
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths inside the Conda environment
DEFAULT_MODEL_NAME = "nanobodies_lstm_sequence_model_trained"
DEFAULT_CHAR_MAPPING_NAME = "char_mapping.csv"
DEFAULT_MODEL_PATH = os.path.join(PACKAGE_DIR, DEFAULT_MODEL_NAME)
DEFAULT_CHAR_MAPPING_PATH = os.path.join(PACKAGE_DIR, DEFAULT_CHAR_MAPPING_NAME)
SEQUENCE_LENGTH = 150

def load_character_mapping(char_mapping_path):
    char_mapping_df = pd.read_csv(char_mapping_path)
    return dict(zip(char_mapping_df["chars"], char_mapping_df["integers"]))

def tokenize_sequence(sequence, char_map):
    return [char_map[ch] for ch in sequence]

def trim_padding(input_seq, pred_seq):
    trimmed_input = input_seq.rstrip("Z")
    trimmed_pred = pred_seq[:len(trimmed_input)]
    return trimmed_input, trimmed_pred

# Function to extract the hidden‐state values
def process_hidden_states(input_file, output_file, model_path, char_mapping_path):
    print(f"Receiving model path {model_path}")
    model = tf.keras.models.load_model(model_path)
    char_to_idx = load_character_mapping(char_mapping_path)

    df = pd.read_csv(input_file)
    if "input" not in df.columns:
        raise ValueError("The input CSV file must contain a column named 'input'.")
    if "identifier" not in df.columns:
        raise ValueError("The input CSV file must contain a column named 'identifier'.")
    print(f"Successfully read input file {input_file}")

    df["input_padded"] = df["input"].apply(lambda seq: seq.ljust(SEQUENCE_LENGTH, "Z"))
    tokenized_sequences = np.array([tokenize_sequence(seq, char_to_idx) for seq in df["input_padded"]])
    x_array = tokenized_sequences.astype("int32")
    print("Successfully tokenized input data")

    PAD_TOKEN = char_to_idx["Z"]
    all_tokens = x_array[x_array != PAD_TOKEN]

	# re-build the model layer by layer
	# so one can extract the hidden states of the last layer
	# i.e. the output of the last LSTM layer
    embedding_layer = model.layers[0]
    bilstm1 = model.layers[1]
    bilstm2 = model.layers[2]
    input_tensor = model.input
    x_emb = embedding_layer(input_tensor)
    x_bi1 = bilstm1(x_emb)
    x_bi2 = bilstm2(x_bi1)
    extract_hs_model = Model(inputs=input_tensor, outputs=x_bi2)
    hidden_states = extract_hs_model(x_array)
    hidden_states_np = hidden_states.numpy()

	# prepare containers to save the hidden states and metadata
    all_hidden_states = []
    all_positions = []
    all_pred_classes = []
    all_sequence_ids = []

    num_samples, _, _ = hidden_states_np.shape
    identifier_list = df["identifier"].tolist()

    print("Initiating hidden states extraction")
    for seq_id in range(num_samples):
        identifier = identifier_list[seq_id]
        for pos in range(SEQUENCE_LENGTH):
            if x_array[seq_id, pos] != PAD_TOKEN:
                all_hidden_states.append(hidden_states_np[seq_id, pos, :])
                all_positions.append(pos)
                all_pred_classes.append(np.argmax(model.predict(x_array)[seq_id, pos]))
                all_sequence_ids.append(identifier)

    all_hidden_states = np.array(all_hidden_states)
    all_positions = np.array(all_positions)
    all_pred_classes = np.array(all_pred_classes)
    all_sequence_ids = np.array(all_sequence_ids)
    N = len(all_tokens)
    assert all_hidden_states.shape[0] == N
    assert len(all_positions) == N
    assert len(all_pred_classes) == N
    assert len(all_sequence_ids) == N
    print("Hidden states extraction")

    df_hidden_states = pd.DataFrame({
        'position': all_positions,
        'pred_class': all_pred_classes,
        'sequence_id': all_sequence_ids,
        'token': all_tokens
    })

    hidden_dim = all_hidden_states.shape[1]
    for d in range(hidden_dim):
        df_hidden_states[f'hs_dim_{d}'] = all_hidden_states[:, d]

    df_hidden_states.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

@click.command()
@click.option('-i', '--input',       'input_file',       type=click.Path(exists=True), default=None,
				help="Path to input CSV (mandatory columns: 'input', 'identifier')")
@click.option('-o', '--output',      'output_file',      type=click.Path(),          default=None,
				help="Path to output CSV where hidden states values will be saved")
@click.option('--model',             'model_path',       type=click.Path(),          default=DEFAULT_MODEL_PATH,
				help="optional path to the trained model (defaults to conda environment assets)")
@click.option('--char_mapping',      'char_mapping_path',type=click.Path(),          default=DEFAULT_CHAR_MAPPING_PATH,
				help="optional path to the character mapping CSV (defaults to conda environment assets)")
def main(input_file, output_file, model_path, char_mapping_path):
    print("Initiating explainability through Saliency Maps with Gradient Tape")
    process_hidden_states(input_file, output_file, model_path, char_mapping_path)

if __name__ == '__main__':
    main()