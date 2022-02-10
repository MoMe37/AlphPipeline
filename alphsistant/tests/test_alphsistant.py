import numpy.testing as npt
import alphsistant as alp
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

def test_cutome_dataset():
    sk_weights = pd.read_csv("./alphsistant/data/ds_weights.csv")
    dataset = alp.CustomSKDataset(sk_weights)
    X, y = dataset[0]
