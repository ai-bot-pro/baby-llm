from modules.VAE.VQVAE.base import VQVAEModelTrainArgs
from modules.VAE.VQVAE.model import VQVAEModel, VQVAEModelArgs
from modules.VAE.VQVAE.model_from_sonnet import VQVAEModel as SonnetVQVAEModel
from modules.VAE.VQVAE.model_from_sonnet import VQVAEModelArgs as SonnetVQVAEModelArgs


vq_vae_models = {
    "VQVAEModel": VQVAEModel,
    "VQVAEModelFromSonnet": SonnetVQVAEModel,
}

vq_vae_default_model_args = {
    "VQVAEModel": VQVAEModelArgs(
        input_dim=3,
        n_embeddings=512,
        hidden_dim=512,
        latent_dim=16,
        output_dim=3,
        commitment_beta=0.25,
    ),
    "VQVAEModelFromSonnet": SonnetVQVAEModelArgs(
        in_channels=3,  # in_channels/out_channels
        embedding_dim=64,
        num_embeddings=512,
        img_size=64,  # w:32,h:32
        beta=0.25,
    ),
}


vq_vae_default_model_train_args = {
    "VQVAEModel": VQVAEModelTrainArgs(),
    "VQVAEModelFromSonnet": VQVAEModelTrainArgs(),
}
