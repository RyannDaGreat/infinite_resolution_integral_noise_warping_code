"""
Numpy implementation of regaussianize() from noise_warp.py.

After nearest-neighbor backward warp, duplicate pixels (many dests reading
the same source) break the Gaussian distribution. Regaussianization restores
it by: (1) grouping pixels by identical channel-0 values, (2) dividing each
group by sqrt(count), and (3) adding zero-sum foreign noise within each group.

Data layout: CHW (channel-first), matching the original torch implementation.
"""

import numpy as np


def unique_pixels(image):
    """
    Pure function. Find groups of identical pixel values in a CHW image.

    Uses only the values to group — identical pixel vectors map to the same group.

    Args:
        image (np.ndarray): [C, H, W] float array.

    Returns:
        tuple: (unique_colors [U, C], counts [U], index_matrix [H, W])

    Examples:
        >>> img = np.array([[[1, 1, 2], [2, 3, 3]]], dtype=np.float32)  # [1, 2, 3]
        >>> uc, counts, idx = unique_pixels(img)
        >>> sorted(counts.tolist())
        [1, 2, 2]
    """
    c, h, w = image.shape
    pixels = image.transpose(1, 2, 0).reshape(-1, c)  # [H*W, C]

    unique_colors, inverse, counts = np.unique(
        pixels, axis=0, return_inverse=True, return_counts=True
    )
    index_matrix = inverse.reshape(h, w)

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Pure function. Sum pixel values within each group defined by index_matrix.

    Args:
        image (np.ndarray): [C, H, W] float array.
        index_matrix (np.ndarray): [H, W] int array with group indices in [0, U).

    Returns:
        np.ndarray: [U, C] summed values per group.

    Examples:
        >>> img = np.ones((2, 3, 3), dtype=np.float32)
        >>> idx = np.zeros((3, 3), dtype=np.int64)
        >>> sum_indexed_values(img, idx)
        array([[9., 9.]], dtype=float32)
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1
    pixels = image.transpose(1, 2, 0).reshape(-1, c)  # [H*W, C]
    flat_idx = index_matrix.ravel()

    output = np.zeros((u, c), dtype=pixels.dtype)
    np.add.at(output, flat_idx, pixels)

    return output


def indexed_to_image(index_matrix, unique_colors):
    """
    Pure function. Reconstruct a CHW image from index_matrix and per-group values.

    Args:
        index_matrix (np.ndarray): [H, W] int array with group indices.
        unique_colors (np.ndarray): [U, C] values per group.

    Returns:
        np.ndarray: [C, H, W] image.

    Examples:
        >>> idx = np.array([[0, 1], [1, 0]])
        >>> colors = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> indexed_to_image(idx, colors).shape
        (2, 2, 2)
    """
    h, w = index_matrix.shape
    flat = unique_colors[index_matrix.ravel()]  # [H*W, C]
    return flat.reshape(h, w, -1).transpose(2, 0, 1)  # [C, H, W]


def regaussianize(noise):
    """
    Restore Gaussian distribution after nearest-neighbor warp.

    Duplicate pixels (from expansion) form groups with identical channel-0 values.
    For each group of size n:
      - Divide original noise by sqrt(n) (correct inflated variance)
      - Add zero-sum foreign noise (break duplicates apart while preserving group sum)

    Result: every pixel is independent N(0,1) again. Proof: if there are no
    duplicates, counts=1 everywhere and noise passes through unchanged. If there
    are duplicates, dividing by sqrt(n) and adding zero-sum noise restores both
    the variance and the independence.

    Not pure: calls np.random.randn internally.

    Args:
        noise (np.ndarray): [C, H, W] float32 after NN warp.

    Returns:
        tuple: (output [C, H, W] float32, counts_image [1, H, W] float32)

    Examples:
        >>> n = np.random.randn(4, 8, 8).astype(np.float32)
        >>> out, counts = regaussianize(n)
        >>> out.shape
        (4, 8, 8)
    """
    c, h, w = noise.shape

    # Group by channel 0 — identical ch0 values ↔ came from same source pixel
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)

    # Foreign noise: independent N(0,1) for each pixel
    foreign_noise = np.random.randn(c, h, w).astype(noise.dtype)

    # Mean of foreign noise within each group → subtract to get zero-sum
    summed = sum_indexed_values(foreign_noise, index_matrix)  # [U, C]
    meaned = summed / counts[:, None]  # [U, C]
    meaned_image = indexed_to_image(index_matrix, meaned)  # [C, H, W]
    zeroed_foreign_noise = foreign_noise - meaned_image

    # Counts per pixel as a [1, H, W] image
    counts_image = indexed_to_image(index_matrix, counts[:, None].astype(noise.dtype))

    # Divide by sqrt(count) to correct variance, add zero-sum noise
    output = noise / np.sqrt(counts_image)
    output = output + zeroed_foreign_noise

    return output, counts_image
