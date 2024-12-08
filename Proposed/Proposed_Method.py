import json
import base64
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import wave
import pydicom
import numpy as np
import wave
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

# Function to generate salt
def generate_salt():
    return os.urandom(16)

# Function to derive key from password
def derive_key(password, salt):
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1, backend=default_backend())
    return kdf.derive(password.encode())

# Function to encrypt key data
def encrypt_key(key_data, password):
    salt = generate_salt()
    key = derive_key(password, salt)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # JSON encode and pad data
    padder = padding.PKCS7(128).padder()  # AES block size is 128 bits
    padded_data = padder.update(json.dumps(key_data).encode()) + padder.finalize()

    encrypted_key = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(salt + iv + encrypted_key).decode()

# Function to decrypt key data
def decrypt_key(encrypted_key, password):
    encrypted_key = base64.b64decode(encrypted_key)
    salt = encrypted_key[:16]
    iv = encrypted_key[16:32]
    encrypted_data = encrypted_key[32:]

    key = derive_key(password, salt)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt and unpad data
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()

    return json.loads(unpadded_data.decode())

# Function to load DICOM file
def load_dicom(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        return dicom
    except Exception as e:
        print(f"Error loading DICOM file: {e}")
        return None

# Function to extract pixel data from DICOM
def extract_pixel_data(dicom):
    try:
        pixel_data = dicom.pixel_array
        return pixel_data
    except Exception as e:
        print(f"Error extracting pixel data: {e}")
        return None

# Function to load audio file
def load_audio(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        params = audio_file.getparams()
        frames = audio_file.readframes(audio_file.getnframes())
    return frames, params

# Function to convert audio data to binary
def audio_to_binary(audio_data):
    binary_audio = ''.join(format(byte, '08b') for byte in audio_data)
    return binary_audio

# Function to embed audio in DICOM using Adaptive Pixel Value Differencing
def embed_audio_in_dicom(pixel_data, audio_binary):
    idx = 0
    embedding_map = []
    height, width = pixel_data.shape
    pixel_data = pixel_data.astype(np.int32)

    for i in range(height):
        for j in range(width - 1):
            if idx < len(audio_binary):
                pixel_diff = abs(pixel_data[i, j] - pixel_data[i, j + 1])
                if pixel_diff >= 16 and idx + 2 <= len(audio_binary):
                    pixel_data[i, j] = (pixel_data[i, j] & ~0b11) | int(audio_binary[idx:idx+2], 2)
                    embedding_map.append((i, j, 2))
                    idx += 2
                elif pixel_diff >= 8 and idx < len(audio_binary):
                    pixel_data[i, j] = (pixel_data[i, j] & ~0b1) | int(audio_binary[idx], 2)
                    embedding_map.append((i, j, 1))
                    idx += 1
            if idx >= len(audio_binary):
                break
        if idx >= len(audio_binary):
            break

    print("Audio data embedded successfully using Proposed Method.")
    return pixel_data, embedding_map

# Function to save the modified DICOM file
def save_dicom(dicom, pixel_data, file_path):
    try:
        dicom.PixelData = pixel_data.astype(np.uint16).tobytes()  # Ensure the data type matches DICOM format
        dicom.save_as(file_path)
    except Exception as e:
        print(f"Error saving modified DICOM file: {e}")

# Function to extract audio from DICOM
def extract_audio_from_dicom(pixel_data, key):
    audio_binary = []
    embedding_map = key['embedding_map']
    idx = 0

    for entry in embedding_map:
        i, j, bits = entry
        if bits == 2:
            audio_binary.append(format(pixel_data[i, j] & 0b11, '02b'))
            idx += 2
        elif bits == 1:
            audio_binary.append(format(pixel_data[i, j] & 0b1, '01b'))
            idx += 1

    audio_binary_str = ''.join(audio_binary)
    audio_bytes = bytearray(int(audio_binary_str[i:i+8], 2) for i in range(0, len(audio_binary_str), 8))
    return audio_bytes

# Function to save the encrypted key
def save_encrypted_key(key, key_file, password):
    encrypted_key = encrypt_key(key, password)  # Encrypt the key with a password
    with open(key_file, 'w') as f:
        f.write(encrypted_key)

# Function to load and decrypt the key
def load_decrypted_key(key_file, password):
    with open(key_file, 'r') as f:
        encrypted_key = f.read()
    key = decrypt_key(encrypted_key, password)
    return key

# Function to save extracted audio to file
def save_extracted_audio(audio_data, params, output_file):
    with wave.open(output_file, 'wb') as audio_file:
        audio_file.setnchannels(params.nchannels)
        audio_file.setsampwidth(params.sampwidth)
        audio_file.setframerate(params.framerate)
        audio_file.writeframes(audio_data)

# Function to plot time taken for embedding and extraction
def plot_insertion_extraction_times(embed_times, extract_times, audio_files):
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


def main():
    dicom_file = 'input4.dcm'
    audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav','audio4.wav','audio5.wav','audio6.wav']  # List of audio files
    extracted_audio_files = ['extracted_audio1.wav', 'extracted_audio2.wav', 'extracted_audio3.wav','extracted_audio4.wav','extracted_audio5.wav','extracted_audio6.wav']
    key_file_template = 'embedding_key_{}.json'  # Template for key file names
    password = 'strongpassword'

    dicom = load_dicom(dicom_file)
    pixel_data = extract_pixel_data(dicom)

    insertion_times = []
    extraction_times = []

    for i, audio_file in enumerate(audio_files):
        audio_data, params = load_audio(audio_file)
        audio_binary = audio_to_binary(audio_data)

        print(f"Processing audio file: {audio_file}")

        start_insertion = time.time()
        embedded_pixel_data, embedding_map = embed_audio_in_dicom(pixel_data, audio_binary)
        output_dicom_file = f'output_{audio_file.split(".")[0]}.dcm'
        save_dicom(dicom, embedded_pixel_data, output_dicom_file)
        insertion_time = time.time() - start_insertion

        key_data = {'audio_length': len(audio_binary), 'embedding_map': embedding_map}
        key_file = key_file_template.format(i+1)  # Generate unique key file name
        save_encrypted_key(key_data, key_file, password)

        # Decrypt the key
        decrypted_key = load_decrypted_key(key_file, password)

        start_extraction = time.time()
        extracted_audio = extract_audio_from_dicom(embedded_pixel_data, decrypted_key)
        extracted_audio_file = extracted_audio_files[i]
        save_extracted_audio(extracted_audio, params, extracted_audio_file)
        extraction_time = time.time() - start_extraction

        # Print the time taken for insertion and extraction
        print(f"Insertion time: {insertion_time:.4f} seconds")
        print(f"Extraction time: {extraction_time:.4f} seconds")
        
        insertion_times.append(insertion_time)
        extraction_times.append(extraction_time)

    # Plot insertion and extraction times for each audio file
    plot_insertion_extraction_times(insertion_times, extraction_times, audio_files)

# Run the main process
if __name__ == "__main__":
    main()