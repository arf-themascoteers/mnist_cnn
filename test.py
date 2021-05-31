import numpy as np
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn
import torch
from torch import optim
import net
import torch.nn.functional as F

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

valset = datasets.MNIST('data', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = net.Net()
model.load_state_dict(torch.load("models/cnn.h5"))
model.eval()

with torch.no_grad():
    for images, labels in valloader:
        output = model(images)
        pred = output.data.max(1, keepdim=True)[1]
        print(F.nll_loss(output, labels, size_average=True).item())
        print("Total ",pred.shape[0])
        print("Correct ",pred.eq(labels.data.view_as(pred)).sum().item())

print("done")


