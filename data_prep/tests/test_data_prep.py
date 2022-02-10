import data_prep.data_prep as dp
import numpy.testing as npt

def test_mel_spec():
    audio_path = "../AlphData/fadg0/audio/sa1.wav"
    mel_spec = dp.mel_spec(audio_path, 119)
    npt.assert_equal(len(mel_spec), 119)
    npt.assert_equal(len(mel_spec[0]), 128)
    npt.assert_equal(len(mel_spec[0][0]), 32)
    npt.assert_equal(len(mel_spec[118][0]), 32)

def test_spectrogram_extraction():
    dp.spectrogram_extraction('sa1')
    liste = ['sa1', 'sa2', 'si649', 'si1279', 'si1909', 'sx19', 'sx109', 'sx199', 'sx289', 'sx379']
    for audio_file in liste:
        dp.spectrogram_extraction(audio_file)