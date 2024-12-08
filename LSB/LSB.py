import pydicom
import numpy as np
import wave
import time  # To measure time taken by embedding and extraction
import matplotlib.pyplot as plt  # To plot bar graphs
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Function to calculate PSNR
def calculate_psnr(original_image, modified_image):
    mse = np.mean((original_image - modified_image) ** 2)
    if mse == 0:  # No noise is present in the signal
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Function to calculate SSIM
def calculate_ssim(original_image, modified_image):
    ssim_value = ssim(original_image, modified_image, data_range=modified_image.max() - modified_image.min())
    return ssim_value

# Function to load DICOM file
def load_dicom(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        print("DICOM file loaded successfully.")
        return dicom
    except Exception as e:
        print(f"Error loading DICOM file: {e}")
        return None

# Function to load audio file
def load_audio(file_path):
    try:
        with wave.open(file_path, 'rb') as audio_file:
            params = audio_file.getparams()  # Get audio metadata
            frames = audio_file.readframes(audio_file.getnframes())
        print(f"Audio file loaded successfully. Duration: {audio_file.getnframes() / params.framerate} seconds.")
        return frames, params
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

# Function to convert audio data to binary
def audio_to_binary(audio_data):
    binary_audio = ''.join(format(byte, '08b') for byte in audio_data)
    print("Audio data converted to binary.")
    return binary_audio

# Function to extract pixel data from DICOM
def extract_pixel_data(dicom):
    try:
        pixel_data = dicom.pixel_array
        print("DICOM pixel data extracted successfully.")
        return pixel_data
    except Exception as e:
        print(f"Error extracting pixel data: {e}")
        return None

# Function to embed audio data into DICOM pixel data using LSB
def embed_audio_in_dicom(pixel_data, audio_binary, original_shape):
    start_time = time.time()  # Start time for embedding

    if len(audio_binary) > len(pixel_data) * 8:
        print("Error: Audio data is too large to embed.")
        return None, 0

    pixel_binary = [format(pixel, '08b') for pixel in pixel_data]

    for i in range(len(audio_binary)):
        pixel_binary[i] = pixel_binary[i][:-1] + audio_binary[i]

    embedded_pixel_data = [int(pixel, 2) for pixel in pixel_binary]
    embedded_pixel_data = np.array(embedded_pixel_data, dtype=pixel_data.dtype).reshape(original_shape)

    end_time = time.time()  # End time for embedding
    embedding_time = end_time - start_time
    print(f"Audio data embedded successfully. Time taken: {embedding_time:.4f} seconds.")
    
    return embedded_pixel_data, embedding_time

# Function to save the modified DICOM file
def save_dicom(dicom, pixel_data, file_path):
    try:
        dicom.PixelData = pixel_data.tobytes()
        dicom.save_as(file_path)
        print(f"Modified DICOM file saved as {file_path}")
    except Exception as e:
        print(f"Error saving modified DICOM file: {e}")

# Function to extract audio from the DICOM pixel data
def extract_audio_from_dicom(pixel_data, audio_length):
    start_time = time.time()  # Start time for extraction
    
    pixel_binary = [format(pixel, '08b') for pixel in pixel_data.flatten()]
    extracted_bits = ''.join([pixel[-1] for pixel in pixel_binary[:audio_length * 8]])
    extracted_audio_bytes = int(extracted_bits, 2).to_bytes(len(extracted_bits) // 8, byteorder='big')

    end_time = time.time()  # End time for extraction
    extraction_time = end_time - start_time
    print(f"Audio data extracted from DICOM pixel data. Time taken: {extraction_time:.4f} seconds.")
    
    return extracted_audio_bytes, extraction_time

# Function to extract and save audio to a WAV file
def extract_and_save_audio(dicom_file, output_audio_file, audio_length, params):
    dicom = load_dicom(dicom_file)
    if dicom is None:
        return 0

    pixel_data = extract_pixel_data(dicom)
    if pixel_data is None:
        return 0

    flattened_pixel_data = pixel_data.flatten()
    extracted_audio_bytes, extraction_time = extract_audio_from_dicom(flattened_pixel_data, audio_length)

    # Save the extracted audio to a WAV file
    with wave.open(output_audio_file, 'wb') as audio_file:
        audio_file.setnchannels(params.nchannels)
        audio_file.setsampwidth(params.sampwidth)
        audio_file.setframerate(params.framerate)
        audio_file.writeframes(extracted_audio_bytes)

    print(f"Extracted audio saved to {output_audio_file}")
    return extraction_time

# Function to plot time taken for embedding and extraction
def plot_time_taken(embed_times, extract_times, audio_files):
    x = np.arange(len(audio_files))  # Set x-axis positions

    # Creating the bar plot
    width = 0.35  # Width of the bars
    bars1 = plt.bar(x - width/2, embed_times, width, label='Embedding', color='blue')
    bars2 = plt.bar(x + width/2, extract_times, width, label='Extraction', color='red')

    # Adding labels and title
    plt.xlabel('Audio Files')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Insertion and Extraction Times')
    plt.xticks(x, audio_files)  # Set the x-ticks to audio file names
    plt.legend()  # Show the legend

    # Annotating each bar with the time taken
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')

    # Display the plot
    plt.tight_layout()
    plt.show()

# Main embedding process
def main():
    dicom_file = 'input4.dcm'
    audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav','audio4.wav','audio5.wav','audio6.wav']  # List of audio files
    extracted_audio_files = ['extracted_audio1.wav', 'extracted_audio2.wav', 'extracted_audio3.wav','extracted_audio4.wav','extracted_audio5.wav','extracted_audio6.wav']

    # Load DICOM file
    dicom = load_dicom(dicom_file)
    if dicom is None:
        return

    # Extract DICOM pixel data
    pixel_data = extract_pixel_data(dicom)
    if pixel_data is None:
        return

    original_shape = pixel_data.shape
    flattened_pixel_data = pixel_data.flatten()

    embed_times = []
    extract_times = []

    for audio_file, extracted_audio_file in zip(audio_files, extracted_audio_files):
        # Load audio file
        audio_data, params = load_audio(audio_file)
        if audio_data is None or params is None:
            continue

        # Convert audio data to binary
        audio_binary = audio_to_binary(audio_data)

        # Embed audio into pixel data
        embedded_pixel_data, embed_audio_time = embed_audio_in_dicom(flattened_pixel_data, audio_binary, original_shape)

        # Save the modified DICOM file with a unique name
        output_dicom_file = f'output_{audio_file.split(".")[0]}.dcm'
        save_dicom(dicom, embedded_pixel_data, output_dicom_file)

        # Extract the audio from the modified DICOM file
        extraction_time = extract_and_save_audio(output_dicom_file, extracted_audio_file, len(audio_data), params)

        embed_times.append(embed_audio_time)
        extract_times.append(extraction_time)
        # Calculate PSNR and SSIM between original and embedded pixel data
        psnr_value = calculate_psnr(pixel_data, embedded_pixel_data)
        ssim_value = calculate_ssim(pixel_data, embedded_pixel_data)

        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")

    # Plot the time taken for embedding and extraction
    plot_time_taken(embed_times, extract_times, audio_files)

# Run the main process
if __name__ == "__main__":
    main()