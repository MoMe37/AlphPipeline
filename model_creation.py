import pandas as pd
import random
import csv
import numpy as np

import os

if __name__ == "__main__": 
    phonemes = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2021/ALphSistant/data_3_1.csv")
    sk_weights = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/ds_weights.csv")
    
    phonemes['ref'] = phonemes['record'] + '/' + phonemes['frame']
    sk_weights['ref'] = sk_weights['sequence'] + '/' + sk_weights['frame']
    phonemes.drop('frame', axis=1)

    data = pd.merge(phonemes, sk_weights)

    data.to_csv('sk_data.csv')

    sk_dataset = CustomSKDataset('sk_data.csv')

    dataloader = DataLoader(sk_dataset, batch_size=512, shuffle=True, num_workers=0)

    layer1 = nn.Linear(in_features=33, out_features=3000)
    layer2 = nn.Linear(in_features=3000, out_features=3000)
    model = nn.Sequential(
        layer1,
        nn.Tanh(),
        layer2,
        nn.Tanh(),
        layer2,
        nn.Tanh(),
        nn.Linear(3000, 8)
    )

    print("Model structure: ", model, "\n\n")

    learning_rate = 1e-3
    batch_size = 64

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer)
        test_loop(dataloader, model, loss_fn)
    print("Done!")

    torch.save(model, 'C:/Users/Enzo.Magal/Documents/Enzo2021/models/sk_model.pth')