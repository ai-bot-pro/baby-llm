import multiprocessing
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt

from modules.VAE.GaussianMLPVAE.model import GaussianMLPVAEModel, DEVICE


def load_data(dataset_path: str, batch_size: int):
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
    print("train_dataset", train_dataset, "test_dataset", test_dataset)

    kwargs = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True if DEVICE == "cuda" else False,
    }
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    print("train_loader", train_loader, "test_loader", test_loader)
    return train_loader, test_loader


def show_image(x, idx, batch_size):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())


if __name__ == "__main__":
    dataset_path = "./datas/datasets"
    os.makedirs(dataset_path, exist_ok=True)
    model_ckpt_path = "./datas/models/VAE/"
    os.makedirs(model_ckpt_path, exist_ok=True)

    # Model Hyperparameters
    batch_size = 100
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200
    learning_rate = 1e-3

    # load dataset
    train_loader, test_loader = load_data(dataset_path=dataset_path, batch_size=batch_size)

    # model
    model = GaussianMLPVAEModel(x_dim, hidden_dim, latent_dim).to(DEVICE)
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model)
    print(model_million_params, "M parameters")
    model_name = "GaussianMLPVAEModel"
    model_id = (
        f"loss_{model_name}_BA:{batch_size}_PAR:{model_million_params:.2f}_LR:{learning_rate}_MNIST"
    )
    model_filename = model_id + ".pth"

    # training
    print("Start training VAE...")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    epochs = 100
    minloss = 100  # Track minimum validation loss found so far.
    for epoch in range(epochs):
        batch_loss = 0
        # len(train_loader) = (Number of datapoints)/batch_size
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim).to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = model.loss_function(x, x_hat, mean, log_var)
            # print(batch_idx, loss)

            batch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = batch_loss / (len(train_loader) * batch_size)
        print("Epoch", epoch + 1, "complete!", "\tAverage Loss: ", avg_loss)

        best_so_far = avg_loss < minloss
        if best_so_far:  # save model ckpt
            # generate from the model
            path = os.path.join(model_ckpt_path, model_filename)
            torch.save(model.state_dict(), path)
            print("Saving model to path", path)

    print("Train Finish!!")

    # generate/sample
    model.eval()
    # generate from pre-trained model with test datasets
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x_hat = model.generate(x, batch_size)
        show_image(x, idx=batch_idx, batch_size=batch_size)
        show_image(x_hat, idx=batch_idx, batch_size=batch_size)

    # sample from decoder with noise vector
    generated_images = model.sample(batch_size)
    show_image(generated_images, idx=12, batch_size=batch_size)
