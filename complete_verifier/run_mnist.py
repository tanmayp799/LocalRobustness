"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.utils import Flatten
import argparse


def mnist_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model
def main():
    parser = argparse.ArgumentParser(description="Run local search on the mnist model. Run with -h for more help")
    
    # Add an argument to select the module
    parser.add_argument(
        '--method', 
        choices=['local', 'global'],  # Allow only specific options
        required=True, 
        help="Select whether to do global or local search"
    )
    parser.add_argument(
        '--k', 
        type=int,  # Expect an integer
        default=1,
        help="Side length of the patch (k)."
    )
    parser.add_argument(
        '--eps', 
        type=float,  # Expect an integer
        default=0.3,
        help="Radius of the epsilon ball"
    )
    # Parse arguments
    args = parser.parse_args()
    k = args.k
    eps = args.eps
    norm = 2
    # Conditionally import the selected module
    if args.method == 'local':
        from auto_LiRPA.perturbations import PerturbationLpNormMasked as pert
        print(f"Local Search with {k=}.")
        ptb = pert(norm = norm, eps = eps, k = k)
        # Your code for module1
    elif args.method == 'global' :
        from auto_LiRPA.perturbations import PerturbationLpNorm as pert
        print("Global Search.")
        ptb = pert(norm = norm, eps = eps)
        # Your code for module2
    model = mnist_model()
    # Optionally, load the pretrained weights.
    checkpoint = torch.load(
        os.path.join(os.path.dirname(__file__), 'pretrained/mnist_a_adv.pth'),
        map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    ## Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=torchvision.transforms.ToTensor())
    # For illustration we only use 2 image from dataset
    N = 50
    n_classes = 10
    image = test_data.data[:N].view(N,1,28,28)
    true_label = test_data.targets[:N]
    # Convert to float
    image = image.to(torch.float32) / 255.0
    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter is for constructing the trace of the computational graph,
    # and its content is not important.
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    print('Running on', image.device)

    toc = time.perf_counter()
    ## Step 4: Compute bounds using LiRPA given a perturbation

    image = BoundedTensor(image, ptb)
    # Get model prediction as usual
    pred = lirpa_model(image)
    label = torch.argmax(pred, dim=1).cpu().detach().numpy()

    ## Step 5: Compute bounds for final output
    methods = ['IBP+backward (CROWN-IBP)', 'backward (CROWN)',
            # 'CROWN-Optimized (alpha-CROWN)'
            ]
    for method in methods:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
        for i in range(N):
            print(f'Image {i} top-1 prediction {label[i]} ground-truth {true_label[i]}')
            for j in range(n_classes):
                indicator = '(ground-truth)' if j == true_label[i] else ''
                print('f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}'.format(
                    j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
        print()
    tic = time.perf_counter()
    time_taken = tic-toc
    print(f"Time taken: {time_taken}")
if __name__ == "__main__":
    main()