from typing import Optional, Union


def compute_heads(model_dim: int, head_dim: int) -> int:
    """Compute the number of heads.

    Args:
        model_dim: Model dimension.
        head_dim: Head dimension.

    ...note:
        If model dimension is not divisible by head dimension, ValueError is raised. Otherwise, integer denoting
        number of heads in multi-head attention is returned.
    """
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    else:
        raise ValueError(
            f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}."
        )


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


if __name__ == "__main__":
    import numpy as np

    # OpenELM-270M Pre-training hyper-parameters
    n_embd = 1280
    n_head = 20
    n_layer = 16
    ffn_intermediate_divisor = 256

    # defualt
    n_embd = 384
    n_head = 6
    n_layer = 6
    ffn_intermediate_divisor = 256
    head_size = n_embd // n_head
    # attention scaling
    qkv_multipliers = [
        round(v, 2)
        for v in np.linspace(
            0.5,
            1.0,
            num=n_layer,
            dtype=float,
        )
    ]
    print(qkv_multipliers)

    query_sizes = [
        int(
            make_divisible(n_embd * m, divisor=head_size)
        )
        for m in qkv_multipliers
    ]
    print(query_sizes)

    num_qkv_heads = [
        int(compute_heads(q_size, head_size)) for q_size in query_sizes
    ]
    print(num_qkv_heads)

    # ffn scaling
    ffn_multipliers = [
        round(v, 2)
        for v in np.linspace(
            0.5,
            4.0,
            num=n_layer,
            dtype=float,
        )
    ]
    print(ffn_multipliers)
    ffn_intermediate_sizes = [
        int(
            make_divisible(n_embd * m, divisor=ffn_intermediate_divisor)
        )
        for m in ffn_multipliers
    ]
    print(ffn_intermediate_sizes)

    """
    # OpenELM-270M Pre-training hyper-parameters
    [0.5, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.77, 0.8, 0.83, 0.87, 0.9, 0.93, 0.97, 1.0]
    [640, 704, 704, 768, 832, 832, 896, 960, 960, 1024, 1088, 1088, 1152, 1216, 1216, 1280]
    [10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20]
    [0.5, 0.73, 0.97, 1.2, 1.43, 1.67, 1.9, 2.13, 2.37, 2.6, 2.83, 3.07, 3.3, 3.53, 3.77, 4.0]
    [768, 1024, 1280, 1536, 1792, 2048, 2560, 2816, 3072, 3328, 3584, 3840, 4352, 4608, 4864, 5120]

    # defualt
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    [192, 256, 256, 320, 320, 384]
    [3, 4, 4, 5, 5, 6]
    [0.5, 1.2, 1.9, 2.6, 3.3, 4.0]
    [256, 512, 768, 1024, 1280, 1536]
    """
