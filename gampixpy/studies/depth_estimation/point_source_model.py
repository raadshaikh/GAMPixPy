import numpy as np

import torch 
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import h5py
import matplotlib.pyplot as plt

class LinearModel (nn.Module):
    def __init__(self, **kwargs):
        super(LinearModel, self).__init__(**kwargs)

        self.model = nn.Sequential(nn.Linear(12, 128),
                                   nn.Linear(128, 128),
                                   # nn.ReLU(),
                                   nn.Linear(128, 128),
                                   # nn.ReLU(),
                                   nn.Linear(128, 128),
                                   # nn.ReLU(),
                                   # nn.Linear(128, 4),
                                   nn.Linear(128, 1),
                                   )

    def forward(self, x):
        return self.model(x)

def loss(truth, inference):
    # truth_pos = truth[:,:3]
    # inf_pos = inference[:,:3]

    mse_loss = torch.mean(torch.pow(truth - inference, 2))
    # position_loss = torch.sum(torch.pow(truth_pos - inf_pos, 2))
    # print ("position loss", position_loss.item())
    
    # truth_charge = truth[:,3]
    # inf_charge = truth[:,3]
    # charge_loss =

    return mse_loss    
    
f = h5py.File('gampixD_summary_metrics.h5')

labels = torch.tensor([f['labels']['vertex x'],
                       f['labels']['vertex y'],
                       f['labels']['vertex z'],
                       f['labels']['deposited charge'],
                       ]).T.to(device)
metrics = torch.tensor([f['metrics']['pixel mean x'],
                        f['metrics']['pixel mean y'],
                        f['metrics']['pixel var x'],
                        f['metrics']['pixel var y'],
                        f['metrics']['pixel var t'],
                        f['metrics']['pixel total q'],
                        f['metrics']['coarse mean x'],
                        f['metrics']['coarse mean y'],
                        f['metrics']['coarse var x'],
                        f['metrics']['coarse var y'],
                        f['metrics']['coarse var t'],
                        f['metrics']['coarse total q'],
                        ]).T.to(device)

batch_size = 256

model = LinearModel().to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1.e-5,
                             # weight_decay=5e-4,
                             )

n_samples = metrics.shape[0]

feature_index = 2

loss_series = []

max_epoch = 100
n_epoch = 0
while n_epoch < max_epoch:

    sample_order = np.random.choice(n_samples,
                                    n_samples,
                                    replace = False)
    cursor = 0
    n_iter = 0
    iter_per_epoch = n_samples//batch_size
    while cursor + batch_size < n_samples:
        batch_labels = labels[cursor:cursor+batch_size,feature_index,None]
        batch_metrics = metrics[cursor:cursor+batch_size, :]

        batch_inference = model(batch_metrics)

        # print (batch_labels.shape, batch_inference.shape)
        mse_loss = loss(batch_labels, batch_inference)

        print (n_iter, iter_per_epoch, mse_loss.item())
        mse_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print (model(batch_metrics.T))

        loss_series.append(mse_loss.item())
        
        cursor += batch_size
        n_iter += 1
        
    n_epoch += 1

batch_size = 8

sample_order = np.random.choice(n_samples,
                                n_samples,
                                replace = False)
cursor = 0
n_iter = 0
iter_per_epoch = n_samples//batch_size
while n_iter < 4:
    batch_labels = labels[cursor:cursor+batch_size, feature_index, None]
    batch_metrics = metrics[cursor:cursor+batch_size, :]

    batch_inference = model(batch_metrics)
    
    mse_loss = loss(batch_labels, batch_inference)

    print (n_iter, iter_per_epoch, mse_loss.item())
    print ("label", batch_labels, batch_inference)
    # print ("diff", (batch_labels - batch_inference)/batch_labels)
    print ("diff", (batch_labels - batch_inference)/batch_labels)
        
    cursor += batch_size
    n_iter += 1
    
fig = plt.figure()
plt.plot(loss_series)
plt.semilogy()

plt.show()
