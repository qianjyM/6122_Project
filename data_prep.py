import matplotlib
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import concurrent.futures
import librosa.display

# Set matplotlib to use a non-interactive backend
matplotlib.use('Agg')

def save_spectrogram(log_mel_spec, sr, file_name, save_path):
    # Saves a log-mel spectrogram as an image file
    plt.ioff()  # Turn off interactive plotting
    fig, ax = plt.subplots(figsize=(4, 3))
    librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='gray', ax=ax)
    ax.axis('off')  # Hide axes
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)  # Adjust subplot to remove whitespace
    plt.savefig(os.path.join(save_path, f"{file_name}_log_mel_spectrogram.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_segment(audio_segment, sr, file_name, save_path, log_file, segment_number):
    # Processes and saves a segment of an audio file
    max_amplitude = np.max(np.abs(audio_segment))  # Normalize amplitude
    if max_amplitude > 0:
        normalized_audio = audio_segment / max_amplitude
    else:
        return  # Skip processing if the segment is silent

    # Generate a mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=normalized_audio, sr=sr, n_fft=512, hop_length=256, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Save the spectrogram as an image file
    segment_file_name = f"{file_name}_segment_{segment_number + 1}"
    save_spectrogram(log_mel_spec, sr, segment_file_name, save_path)
    log_file.write(f"Processed: {segment_file_name}\n")

def process_dog_audio(raw_audio, file_name, total_length, sr, save_path, log_file):
    # Processes dog audio files with overlapping windows
    target_length = int(2.5 * sr)  # Target length of each segment
    overlap_size = int(0.5 * sr)  # Overlap size between segments
    segment_start = 0
    segment_count = 0
    while segment_start + target_length <= total_length:
        segment_end = segment_start + target_length
        audio_segment = raw_audio[segment_start:segment_end]
        process_segment(audio_segment, sr, file_name, save_path, log_file, segment_count)
        segment_start += overlap_size
        segment_count += 1
    # Zero-padding for audio shorter than target length
    if total_length < target_length:
        padding = target_length - total_length
        padded_audio = np.pad(raw_audio, (0, padding), mode='constant')
        process_segment(padded_audio, sr, file_name, save_path, log_file, segment_count)

def process_cat_audio(raw_audio, file_name, total_length, sr, save_path, log_file):
    # Processes cat audio files, splitting or padding them as needed
    target_length = int(2.5 * sr)  # Target length for cat audio segments
    # Zero-padding for audio shorter than target length
    if total_length < target_length:
        padding = target_length - total_length
        raw_audio = np.pad(raw_audio, (padding // 2, padding - padding // 2), mode='constant')
        total_length = target_length
    num_segments = int(np.ceil(total_length / target_length))
    for segment in range(num_segments):
        start_sample = segment * target_length
        end_sample = min(start_sample + target_length, total_length)
        audio_segment = raw_audio[start_sample:end_sample]
        process_segment(audio_segment, sr, file_name, save_path, log_file, segment)

def process_audio_file(file_path, sr, cat_save_path, dog_save_path, cat_log_file, dog_log_file, other_log_file):
    # Processes an audio file based on its filename
    try:
        raw_audio, _ = librosa.load(file_path, sr=sr)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Process based on the file name prefix
        if file_name.lower().startswith('dog'):
            process_dog_audio(raw_audio, file_name, len(raw_audio), sr, dog_save_path, dog_log_file)
        elif file_name.lower().startswith('cat'):
            process_cat_audio(raw_audio, file_name, len(raw_audio), sr, cat_save_path, cat_log_file)
        else:
            other_log_file.write(f"File does not match 'cat' or 'dog': {file_name}\n")
    except Exception as e:
        other_log_file.write(f"Error processing {file_name}: {e}\n")

def main(data_folder_path, cat_save_path, dog_save_path, sr, cat_log_file_name, dog_log_file_name, other_log_file_name):
    # Main function to orchestrate audio processing
    report_directory = os.path.join(data_folder_path, 'Processed', 'Report')
    os.makedirs(cat_save_path, exist_ok=True)  # Ensure the save path for cat images exists
    os.makedirs(dog_save_path, exist_ok=True)  # Ensure the save path for dog images exists
    os.makedirs(report_directory, exist_ok=True)  # Ensure the report directory exists

    # Open log files
    cat_log_file_path = os.path.join(report_directory, cat_log_file_name)
    dog_log_file_path = os.path.join(report_directory, dog_log_file_name)
    other_log_file_path = os.path.join(report_directory, other_log_file_name)

    with open(cat_log_file_path, 'w') as cat_log_file, open(dog_log_file_path, 'w') as dog_log_file, open(other_log_file_path, 'w') as other_log_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file in os.listdir(data_folder_path):
                if file.endswith('.wav'):
                    futures.append(executor.submit(
                        process_audio_file, os.path.join(data_folder_path, file), sr,
                        cat_save_path, dog_save_path, cat_log_file, dog_log_file, other_log_file))
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    data_folder_path = r'F:\Desktop\GT_Study\ECE_6122\Final_Project\Audio_data\cats_dogs'
    cat_save_path = os.path.join(data_folder_path, 'Processed', 'Cats')
    dog_save_path = os.path.join(data_folder_path, 'Processed', 'Dogs')
    sr = 22050
    main(data_folder_path, cat_save_path, dog_save_path, sr, 'cat_process_log.txt', 'dog_process_log.txt', 'other_process_log.txt')
