import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def mel_spec(audio_path, nbr_sample):
    audio, sample_rate = librosa.load(audio_path, sr=None)
    sample = []
    for i in range(nbr_sample):
        sample.append(audio[int(i*(len(audio)/nbr_sample)):int((i+1)*(len(audio)/nbr_sample))])
    mel_spec = []
    for i in range(nbr_sample):
        spectrogram = librosa.stft(sample[i], n_fft = 1024, hop_length = 41, win_length=82)
        sgram_mag, _ = librosa.magphase(spectrogram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr= sample_rate)
        mel_spec.append(librosa.amplitude_to_db(mel_scale_sgram, ref=np.min))
    librosa.display.specshow(mel_spec[nbr_sample-1], sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    return mel_spec

def spectrogram_extraction(record):
    audio_path = "../AlphData/fadg0/audio/" + record + ".wav"
    video_path = "../AlphData/fadg0/video/" + record
    nbr_sample = 0
    for path in os.listdir(video_path):
        if os.path.isfile(os.path.join(video_path, path)):
            nbr_sample += 1
    mel_spectrogram = mel_spec(audio_path, nbr_sample)
    spec_path = "../AlphData/fadg0/spectrogram/" + record
    try: 
        os.makedirs(spec_path)
    except OSError:
        if not os.path.isdir(spec_path):
            Raise
    for filename in os.listdir(spec_path) :
        os.remove(spec_path + "/" + filename)
    for i in range(nbr_sample):
        np.savetxt(spec_path + "/face_" + '{:03}'.format(i+1) + ".txt", mel_spectrogram[i], fmt='%d')
