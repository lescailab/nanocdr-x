import os
import click
import tensorflow as tf
import numpy as np
import pandas as pd

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

# -----------------------------------------------------------------------------
# Compute Gradient-Based Saliency for a Single Sample
# -----------------------------------------------------------------------------
def compute_grad_saliency_for_sample(x_single, padded_token, embedding_layer, sub_model, model):
    nonpad_idx = np.where(x_single[0] != padded_token)[0]
    actual_length = nonpad_idx[-1] + 1 if len(nonpad_idx) else 0

    print("Running predictions to be used for saliency map")
    predictions = model.predict(x_single, verbose=0)[0]
    predicted_classes = np.argmax(predictions, axis=1)[:actual_length]

    saliency_map = np.zeros((actual_length, SEQUENCE_LENGTH), dtype=np.float32)
    x_tensor = tf.constant(x_single)
    print("Computing saliency for each time step")
    for t in range(actual_length):
        c = predicted_classes[t]
        with tf.GradientTape() as tape:
            embedding_output = embedding_layer(x_tensor)
            tape.watch(embedding_output)
            sub_out = sub_model(embedding_output)
            prob_c_t = sub_out[0, t, c]
        grad_embed = tape.gradient(prob_c_t, embedding_output)
        grad_mean = tf.reduce_mean(tf.abs(grad_embed), axis=-1)
        grad_values = grad_mean.numpy()[0]
        saliency_map[t, :] = grad_values
    return saliency_map, predicted_classes, actual_length

def process_saliency(input_file, output_file, model_path, char_mapping_path):
    print(f"Receiving model path {model_path}")
    model = tf.keras.models.load_model(model_path)
    char_to_idx = load_character_mapping(char_mapping_path)

    df = pd.read_csv(input_file)
    if "input" not in df.columns or "identifier" not in df.columns:
        raise ValueError("Input CSV must have 'input' and 'identifier'.")
    print(f"Successfully read input file {input_file}")

    df["input_padded"] = df["input"].apply(lambda seq: seq.ljust(SEQUENCE_LENGTH, "Z"))
    tokenized_sequences = np.array([tokenize_sequence(seq, char_to_idx) for seq in df["input_padded"]])
    x_array = tokenized_sequences.astype("int32")

    PAD_TOKEN = char_to_idx["Z"]
    embedding_layer = model.layers[0]
    sub_model = tf.keras.Sequential(model.layers[1:])

    num_samples = x_array.shape[0]
    identifier_list = df["identifier"].tolist()
    saliency_rows = []

    print("Formatting saliency maps for all samples")
    for i in range(num_samples):
        x_single = x_array[i:i+1]
        sal_map, pred_classes, actual_length = compute_grad_saliency_for_sample(
            x_single, PAD_TOKEN, embedding_layer, sub_model, model
        )
        seq_identifier = identifier_list[i]
        for t in range(actual_length):
            for p in range(actual_length):
                saliency_rows.append({
                    "sequence_id": seq_identifier,
                    "time_point": t + 1,
                    "position":   p + 1,
                    "saliency":   sal_map[t, p],
                    "class":      int(pred_classes[t])
                })
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} samples.")

    df_saliency = pd.DataFrame(saliency_rows)
    df_saliency.to_csv(output_file, index=False)
    print(f"Saved detailed saliency values to {output_file}")

@click.command()
@click.option('-i', '--input',   'input_file',       type=click.Path(exists=True), default=None,
				help="Path to input CSV (mandatory columns: 'input', 'identifier')")
@click.option('-o', '--output',  'output_file',      type=click.Path(),          default=None,
				help="Path to output CSV where saliency values will be saved")
@click.option('--model',         'model_path',       type=click.Path(),          default=DEFAULT_MODEL_PATH,
				help="optional path to the trained model (defaults to conda environment assets)")
@click.option('--char_mapping',  'char_mapping_path',type=click.Path(),          default=DEFAULT_CHAR_MAPPING_PATH,
				help="optional path to the character mapping CSV (defaults to conda environment assets)")
def main(input_file, output_file, model_path, char_mapping_path):
    print("Initiating explainability through Saliency Maps with Gradient Tape")
    process_saliency(input_file, output_file, model_path, char_mapping_path)

if __name__ == '__main__':
    main()