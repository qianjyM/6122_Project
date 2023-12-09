import matplotlib
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import librosa.display
import concurrent.futures

# Set matplotlib to use a non-interactive backend
matplotlib.use('Agg')


def save_spectrogram(log_mel_spec, sr, file_name, save_path):
    # Saves a log-mel spectrogram as an image file
    plt.ioff()  # Turn off interactive plotting
    fig, ax = plt.subplots(figsize=(4, 3))
    img = librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='gray', ax=ax)
    ax.axis('off')  # Hide axes
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)  # Adjust subplot to remove whitespace
    output_path = os.path.join(save_path, f"{file_name}_log_mel_spectrogram.png")
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    return output_path  # Return the path to the saved figure


def process_segment(audio_segment, sr, file_name, save_path, log_file, segment_number):
    # Skip processing if the segment is silent or nearly silent
    max_amplitude = np.max(np.abs(audio_segment))
    if max_amplitude < 1e-4:  # Threshold for silence
        log_file.write(f"Skipping segment {segment_number + 1} - silent or nearly silent\n")
        return None

    # Normalize audio segment
    normalized_audio = audio_segment / max_amplitude
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=normalized_audio, sr=sr, n_fft=512, hop_length=256, n_mels=128)
    # Convert power spectrum to dB scale (logarithmic)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-6)
    # Clip values to a specified dB range to avoid extreme values
    log_mel_spec = np.clip(log_mel_spec, -80, 0)
    # Check if the spectrogram has any dynamic range (not all values are at the boundary)
    if np.all(log_mel_spec == -80) or np.all(log_mel_spec == 0):
        log_file.write(f"Segment {segment_number + 1} produced an empty spectrogram; skipping.\n")
        return None

    # Construct image file path
    segment_file_name = f"{file_name}_segment_{segment_number + 1}"
    image_path = os.path.join(save_path, f"{segment_file_name}_log_mel_spectrogram.png")
    # Save the spectrogram as an image file
    save_spectrogram(log_mel_spec, sr, segment_file_name, save_path)
    # Write to log file that the segment was processed
    log_file.write(f"Processed: {segment_file_name}\n")
    # Return the path of the saved spectrogram image
    return image_path

def circular_padding(audio, target_length):
    if len(audio) >= target_length:
        return audio[:target_length]
    else:
        repeats = target_length // len(audio) + 1
        return np.tile(audio, repeats)[:target_length]

def process_dog_audio(raw_audio, file_name, total_length, sr, save_path, log_file):
    target_length = int(3 * sr)  # Target length of each segment
    overlap_size = int(0.8 * sr)  # Overlap size between segments
    segment_start = 0
    segment_count = 0
    image_paths = []
    while segment_start + target_length <= total_length:
        segment_end = segment_start + target_length
        audio_segment = raw_audio[segment_start:segment_end]
        image_path = process_segment(audio_segment, sr, file_name, save_path, log_file, segment_count)
        if image_path is not None:
            image_paths.append(image_path)
        segment_start += overlap_size
        segment_count += 1

    # Circular padding for audio shorter than target length
    if total_length < target_length:
        padded_audio = circular_padding(raw_audio, target_length)
        image_path = process_segment(padded_audio, sr, file_name, save_path, log_file, segment_count)
        if image_path is not None:
            image_paths.append(image_path)
    return image_paths


def process_cat_audio(raw_audio, file_name, total_length, sr, save_path, log_file):
    target_length = int(3 * sr)  # Target length for cat audio segments
    image_paths = []

    # Check if audio length is less than target length
    if total_length < target_length:
        # Circular padding for audio shorter than target length
        padded_audio = circular_padding(raw_audio, target_length)
    else:
        # Ensure the audio length is a multiple of the target length
        remainder = len(raw_audio) % target_length
        if remainder != 0:
            padding_needed = target_length - remainder
            padded_audio = np.concatenate((raw_audio, raw_audio[:padding_needed]))
        else:
            padded_audio = raw_audio

    num_segments = int(np.ceil(len(padded_audio) / target_length))

    for segment in range(num_segments):
        start_sample = segment * target_length
        end_sample = start_sample + target_length
        audio_segment = padded_audio[start_sample:end_sample]
        image_path = process_segment(audio_segment, sr, file_name, save_path, log_file, segment)
        if image_path is not None:
            image_paths.append(image_path)

    return image_paths



def process_audio_batch(file_paths, sr, cat_save_path, dog_save_path, cat_log_file, dog_log_file, other_log_file):
    image_paths = []
    for file_path in file_paths:
        try:
            raw_audio, _ = librosa.load(file_path, sr=sr)
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            if file_name.lower().startswith('dog'):
                image_paths.extend(
                    process_dog_audio(raw_audio, file_name, len(raw_audio), sr, dog_save_path, dog_log_file))
            elif file_name.lower().startswith('cat'):
                image_paths.extend(
                    process_cat_audio(raw_audio, file_name, len(raw_audio), sr, cat_save_path, cat_log_file))
            else:
                other_log_file.write(f"File does not match 'cat' or 'dog': {file_name}\n")
        except Exception as e:
            other_log_file.write(f"Error processing {file_name}: {e}\n")
    return image_paths


def main(data_folder_path, cat_save_path, dog_save_path, sr, cat_log_file_name, dog_log_file_name, other_log_file_name):
    report_directory = os.path.join(data_folder_path, 'Processed', 'Report')
    os.makedirs(cat_save_path, exist_ok=True)
    os.makedirs(dog_save_path, exist_ok=True)
    os.makedirs(report_directory, exist_ok=True)

    cat_log_file_path = os.path.join(report_directory, cat_log_file_name)
    dog_log_file_path = os.path.join(report_directory, dog_log_file_name)
    other_log_file_path = os.path.join(report_directory, other_log_file_name)

    audio_files = [os.path.join(data_folder_path, file) for file in os.listdir(data_folder_path) if
                   file.endswith('.wav')]

    with open(cat_log_file_path, 'w') as cat_log_file, \
            open(dog_log_file_path, 'w') as dog_log_file, \
            open(other_log_file_path, 'w') as other_log_file, \
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        # Split audio files into batches of 50
        batches = [audio_files[i:i + 10] for i in range(0, len(audio_files), 10)]
        future_to_batch = {
            executor.submit(process_audio_batch, batch, sr, cat_save_path, dog_save_path, cat_log_file, dog_log_file,
                            other_log_file): batch for batch in batches}

        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                image_paths = future.result()
                # Handle the resulting image paths if necessary
            except Exception as e:
                print(f"Error processing batch {batch}: {e}")


if __name__ == "__main__":
    data_folder_path = r'F:\Desktop\GT_Study\ECE_6122\Final_Project\Audio_data\cats_dogs'
    cat_save_path = os.path.join(data_folder_path, 'Processed', 'Cats')
    dog_save_path = os.path.join(data_folder_path, 'Processed', 'Dogs')
    sr = 22050  # Sample rate
    main(data_folder_path, cat_save_path, dog_save_path, sr, 'cat_process_log.txt', 'dog_process_log.txt',
         'other_process_log.txt')
