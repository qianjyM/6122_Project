import zipfile
import librosa
import numpy as np
import concurrent.futures
import os
import noisereduce as nr

# Define the path to the folder containing the cat and dog audio data
data_folder_path = r'C:\Users\erick\Desktop\GT_Study\ECE_6122\Final_Project\Audio_data\cats_dogs'

# Function to process each audio file
def process_audio_file(file_path):
    try:
        # Load audio
        raw_audio, sr = librosa.load(file_path, sr=None)

        # Normalize amplitude
        normalized_audio = librosa.util.normalize(raw_audio)

        # Noise reduction
        denoised_audio = nr.reduce_noise(y=normalized_audio, sr=sr)

        # Feature transformation and extraction
        stft_features = np.abs(librosa.stft(denoised_audio))
        mfcc_features = librosa.feature.mfcc(y=denoised_audio, sr=sr)

        # Data augmentation: pitch shifting
        pitch_shifted_audio = librosa.effects.pitch_shift(normalized_audio, sr=sr, n_steps=2)

        # Data augmentation: adding random noise
        noise = np.random.randn(len(normalized_audio))
        augmented_audio = normalized_audio + 0.005 * noise

        # Return processed features
        return {
            'file_name': os.path.basename(file_path),
            'stft_features': stft_features,
            'mfcc_features': mfcc_features,
            'pitch_shifted_audio': pitch_shifted_audio,
            'augmented_audio': augmented_audio
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Multithreading for processing all files
def process_all_files(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_audio_file, files)

    # Handle the results
    processed_data = [result for result in results if result is not None]
    return processed_data

# Processing all the audio files in the specified folder
processed_dataset = process_all_files(data_folder_path)

# Save processed data and write to a log file
def save_processed_data(data, save_path, log_file_path):
    with open(log_file_path, 'w') as log_file:
        for data_item in data:
            file_name = data_item['file_name'].replace('.wav', '')
            save_file_path = os.path.join(save_path, f"{file_name}_processed.npy")
            np.save(save_file_path, data_item)
            log_file.write(f"Processed and saved: {save_file_path}\n")  # Writing to log file

# Define the save path and log file path
processed_save_path = os.path.join(data_folder_path, 'Processed')
# Update the log file path to be in the 'Processed\report' folder
log_file_path = os.path.join(data_folder_path, 'Processed\\Report\\processed_list.txt')

# Create the 'Processed' and 'report' directories if they do not exist
if not os.path.exists(processed_save_path):
    os.makedirs(processed_save_path)

# Save the processed dataset and write to the log file
save_processed_data(processed_dataset, processed_save_path, log_file_path)
