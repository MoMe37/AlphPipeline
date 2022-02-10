import numpy.testing as npt
import alphsistant as alp
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

def test_mel_spectrogram_creation():
    mel_spec1, sample_rate = alp.mel_spectrogram_creation("../AlphData/fadg0/audio/sa1.wav")

    assert sample_rate == 32000
    assert len(mel_spec1) == 128
    assert len(mel_spec1[0]) == 298

    mel_spec2, sample_rate = alp.mel_spectrogram_creation("../AlphData/fadg0/audio/sa2.wav")

    assert sample_rate == 32000
    assert len(mel_spec2) == 128
    assert len(mel_spec2[0]) == 258

    mel_spec3, sample_rate = alp.mel_spectrogram_creation("../AlphData/fadg0/audio/sx289.wav")

    assert sample_rate == 32000
    assert len(mel_spec3) == 128
    assert len(mel_spec3[0]) == 316
    assert len(mel_spec3[1]) == 316

    npt.assert_almost_equal(len(mel_spec1[0])/119, len(mel_spec2[0])/103, decimal=3)
    npt.assert_almost_equal(len(mel_spec1[0])/119, len(mel_spec3[0])/126, decimal=2)
    npt.assert_almost_equal(len(mel_spec3[0])/126, len(mel_spec2[0])/103, decimal=2)
    npt.assert_almost_equal(len(mel_spec3[0])/126, 2.50, decimal=2)

def test_mel_spec_sample():
    mel_spec1, sample_rate = alp.mel_spectrogram_creation("../AlphData/fadg0/audio/sa1.wav")
    sample = alp.mel_spec_sample(mel_spec1, 119)

    npt.assert_equal(len(sample), 119)
    npt.assert_equal(len(sample[0][0]), 5)
    npt.assert_array_equal(sample[0][:,0], mel_spec1[:,0])
    npt.assert_array_equal(sample[0][:,4], mel_spec1[:,4])
    npt.assert_array_equal(sample[1][:,0], mel_spec1[:,2])
    for i in range(119):
        npt.assert_equal(len(sample[0][i]), 5)
    
    npt.assert_array_equal(sample[118][:,0], mel_spec2[:,293])
    npt.assert_array_equal(sample[118][:,4], mel_spec2[:,297])

    mel_spec2, sample_rate = alp.mel_spectrogram_creation("../AlphData/fadg0/audio/sa2.wav")
    sample = alp.mel_spec_sample(mel_spec2, 103)

    npt.assert_equal(len(sample), 103)
    npt.assert_equal(len(sample[0][0]), 5)
    npt.assert_array_equal(sample[0][:,0], mel_spec2[:,0])
    npt.assert_array_equal(sample[0][:,4], mel_spec2[:,4])
    npt.assert_array_equal(sample[1][:,0], mel_spec2[:,2])


    for i in range(103):
        npt.assert_equal(len(sample[0][i]), 5)

def test_input_data_creation():
    sk_df = pd.read_csv("./alphsistant/data/ds_weights.csv")
    alp.input_data_creation(sk_df)

def test_cutome_dataset():
    print(1)