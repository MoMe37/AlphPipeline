import numpy.testing as npt
import alphsistant as alp
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

def test_cutome_dataset():
    filepath = "./alphsistant/data/ds_weights.csv"
    dataset = alp.CustomSKDataset(filepath)
    X, y = dataset[0]
    npt.assert_equal(len(X), 4)
    npt.assert_equal(len(X[0]), 32)
    npt.assert_equal(len(X[0][0]), 32)
    npt.assert_equal(len(X[3][0]), 32)
    npt.assert_equal(len(y), 8)