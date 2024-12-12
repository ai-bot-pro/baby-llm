import multiprocessing
import os
import time

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.VAE.VQVAE.model import DEVICE, VQVAEModel


def load_data(dataset_path: str, batch_size: int):
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAR10(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = CIFAR10(dataset_path, transform=mnist_transform, train=False, download=True)
    print("train_dataset", train_dataset, "test_dataset", test_dataset)

    kwargs = {
        "num_workers": multiprocessing.cpu_count(),
        "pin_memory": True if DEVICE == "cuda" else False,
    }
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    print("train_loader", train_loader, "test_loader", test_loader)
    return train_loader, test_loader


def draw_sample_image(x, postfix):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.close()


if __name__ == "__main__":
    dataset_path = "./datas/datasets"
    os.makedirs(dataset_path, exist_ok=True)
    model_ckpt_dir = "./datas/models/VAE/"
    os.makedirs(model_ckpt_dir, exist_ok=True)
    gen_data_dir = "./datas/gen/VAE/"
    os.makedirs(gen_data_dir, exist_ok=True)

    # Model Hyperparameters
    batch_size = 128
    input_dim = 3
    n_embeddings = 512
    hidden_dim = 512
    latent_dim = 16
    output_dim = 3
    learning_rate = 2e-4

    img_size = (32, 32)  # (width, height)
    commitment_beta = 0.25

    epochs = 50
    print_step = 50

    # load dataset
    train_loader, test_loader = load_data(dataset_path=dataset_path, batch_size=batch_size)

    # model
    model = VQVAEModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_embeddings=n_embeddings,
        output_dim=output_dim,
    ).to(DEVICE)
    # print the number of parameters in the model
    model_million_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(model, DEVICE)
    print(model_million_params, "M parameters")
    model_name = "VQVAE"
    model_id = f"loss_{model_name}_BA:{batch_size}_PAR:{model_million_params:.2f}_LR:{learning_rate}_CIFAR10"
    model_filename = model_id + ".pth"
    model_ckpt_path = os.path.join(model_ckpt_dir, model_filename)

    if os.path.exists(model_ckpt_path):
        torch.manual_seed(int(time.time() * 1000))
        model.load_state_dict(torch.load(model_ckpt_path, weights_only=True))
        x_hat = model.sample(batch_size)
        draw_sample_image(x_hat, "Random Codes")

    # training
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    epochs = 50
    minloss = 0.2  # Track minimum validation loss found so far.
    print(f"Start training VAE with {epochs} epochs...")
    for epoch in range(epochs):
        batch_loss = 0
        # len(train_loader) = (Number of datapoints)/batch_size
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, commitment_loss, codebook_loss, perplexity = model(x)
            loss = model.loss_function(x, x_hat, commitment_loss, codebook_loss)
            # print(batch_idx, loss)

            batch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = batch_loss / (len(train_loader) * batch_size)
        print("Epoch", epoch + 1, "complete!", "\tAverage Loss: ", avg_loss)

        best_so_far = avg_loss < minloss
        if best_so_far:  # save model ckpt
            # generate from the model
            torch.save(model.state_dict(), model_ckpt_path)
            print("Saving model to path", model_ckpt_path)

    print("Train Finish!!")

    # generate/sample
    model.eval()
    # generate from pre-trained model with test datasets
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.to(DEVICE)
        x_hat = model.generate(x)

        save_image(
            x_hat,
            os.path.join(gen_data_dir, f"test_generated_{batch_idx}.png"),
        )
        draw_sample_image(x[: batch_size // 2], "Ground-truth images")
        draw_sample_image(x_hat[: batch_size // 2], "Reconstructed images")

    # sample from decoder with noise vector
    generated_images = model.sample(batch_size)
    save_image(
        generated_images,
        os.path.join(gen_data_dir, "generated_sample.png"),
    )
    draw_sample_image(generated_images, "Random Generated Images")
