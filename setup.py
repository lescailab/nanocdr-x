from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text()

setup(
    name="nanocdr-x",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.15.1",
        "keras==2.15.0",
        "click==8.1.8",
        "pyreadr==0.5.2"
    ],
    extras_require={
        "macos-arm": ["tensorflow-metal==1.2.0"]  # Only installs on `osx-arm64`
    },
    entry_points={
        "console_scripts": [
            "predict_cdrs=predict_nanobody_cdrs.predict_nanobody_cdrs:main",
            "explain_hidden_states=predict_nanobody_cdrs.explain_hidden_states:main",
            "explain_saliency=predict_nanobody_cdrs.explain_saliency:main",
        ]
    },
    include_package_data=True,
    package_data={
        "predict_nanobody_cdrs": [
            "char_mapping.csv",
            "nanobodies_lstm_sequence_model_trained/saved_model.pb",
            "nanobodies_lstm_sequence_model_trained/keras_metadata.pb",
            "nanobodies_lstm_sequence_model_trained/fingerprint.pb",
            "nanobodies_lstm_sequence_model_trained/variables/variables.index",
            "nanobodies_lstm_sequence_model_trained/variables/variables.data-00000-of-00001",
            "example_data/example_input.csv"
        ]
    },
    exclude_package_data={"": ["*.DS_Store", "nanobodies_lstm_sequence_model_trained/Icon*"]},
    long_description=long_description,
    long_description_content_type="text/markdown"
)