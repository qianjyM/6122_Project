# Audio Data Processing for Animal Sounds

## Overview
This Python script is designed to process audio data, specifically focusing on recognizing and differentiating animal sounds such as cat meows and dog barks. It includes noise reduction, feature extraction, and data augmentation techniques.

## Features
- **Audio Normalization**: Standardizes the amplitude across all recordings for consistency.
- **Noise Reduction**: Implements advanced methodologies using the `noisereduce` library to filter out background noise and other non-essential audio elements.
- **Feature Transformation and Extraction**:
  - Utilizes `librosa` library to perform Short-Time Fourier Transform (STFT) and extract Mel-Frequency Cepstral Coefficients (MFCC) features.
- **Data Augmentation**:
  - Implements pitch shifting to enhance the variability of the dataset.
  - Adds random noise to the audio for robustness.

## Multithreading
- Employs Python's `concurrent.futures` for multithreading to optimize the processing speed.

## Output
- Processed data is saved in `.npy` format, containing extracted features for each audio file.
- A log file is generated to list all processed files with their respective paths.

## Usage
1. Define the path to the folder containing the cat and dog audio data in `data_folder_path`.
2. Run the script to process the audio files in the specified folder.
3. Check the `Processed\Report\processed_list.txt` for a log of processed files.

## Dependencies
- librosa
- numpy
- concurrent.futures
- noisereduce
- os
