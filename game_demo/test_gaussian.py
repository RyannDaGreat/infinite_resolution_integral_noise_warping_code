"""
Statistical validation of the GLSL noise warp output.

Reads noise textures back from GPU, checks:
- Mean ≈ 0 (|mean| < 0.1)
- Std ≈ 1 (0.8 < std < 1.2)
- No NaN/Inf
- Histogram overlay vs N(0,1)
- Anderson-Darling normality test on random pixel subset

Saves validation images to validation/ folder.
"""

import os
import numpy as np
import fire
from scipy import stats


def validate_noise(noise, frame_idx, output_dir="validation"):
    """
    Run statistical checks on a single noise frame.

    Not pure: writes histogram image to disk, prints results.

    Args:
        noise (np.ndarray): Shape [H, W, 4] float32 noise values.
        frame_idx (int): Frame number for labeling.
        output_dir (str): Where to save validation images.

    Returns:
        dict: Keys 'mean', 'std', 'has_nan', 'has_inf', 'ad_pvalue'.

    Examples:
        >>> # validate_noise(np.random.randn(100, 100, 4).astype(np.float32), 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    flat = noise.flatten()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    has_nan = bool(np.any(np.isnan(flat)))
    has_inf = bool(np.any(np.isinf(flat)))

    # Anderson-Darling on random subset (test is slow on large arrays)
    subset = np.random.choice(flat, size=min(5000, len(flat)), replace=False)
    ad_result = stats.anderson(subset, dist="norm", method="interpolate")
    ad_stat = ad_result.statistic
    ad_pvalue = ad_result.pvalue
    ad_pass = ad_pvalue > 0.05

    # Histogram
    _save_histogram(flat, frame_idx, output_dir, mean, std)

    result = dict(
        mean=mean,
        std=std,
        has_nan=has_nan,
        has_inf=has_inf,
        ad_statistic=ad_stat,
        ad_pvalue=ad_pvalue,
        ad_pass=ad_pass,
    )

    status = "PASS" if (
        abs(mean) < 0.1
        and 0.8 < std < 1.2
        and not has_nan
        and not has_inf
    ) else "FAIL"

    print(
        "Frame %d: %s  mean=%.4f std=%.4f nan=%s inf=%s AD=%.2f (p=%.3f %s)"
        % (
            frame_idx, status, mean, std, has_nan, has_inf,
            ad_stat, ad_pvalue, "pass" if ad_pass else "FAIL",
        )
    )
    return result


def _save_histogram(flat, frame_idx, output_dir, mean, std):
    """
    Save histogram of noise values overlaid with N(0,1) PDF.

    Not pure: writes PNG to disk.

    Args:
        flat (np.ndarray): Flattened noise values.
        frame_idx (int): Frame number.
        output_dir (str): Output directory.
        mean (float): Sample mean.
        std (float): Sample std.

    Examples:
        >>> # _save_histogram(np.random.randn(10000), 0, "validation", 0.0, 1.0)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flat, bins=200, density=True, alpha=0.7, label="Shader noise")

    x = np.linspace(-5, 5, 500)
    ax.plot(x, stats.norm.pdf(x), "r-", lw=2, label="N(0,1)")

    ax.set_title(
        "Frame %d — mean=%.4f std=%.4f" % (frame_idx, mean, std)
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(-5, 5)

    path = os.path.join(output_dir, "histogram_frame_%03d.png" % frame_idx)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  Saved", path)


def run_validation():
    """
    Run the demo for a few frames and validate noise statistics.

    Not pure: creates pygame window, runs GPU pipeline, writes files.

    Examples:
        >>> # run_validation()
    """
    import pygame
    import glm

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True
    )
    W, H = 800, 600
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)

    # Import after pygame/GL context is created
    from renderer import Renderer
    from main import SpinningCube, FPSCamera

    renderer = Renderer(W, H)
    camera = FPSCamera()
    cube = SpinningCube()
    proj = glm.perspective(glm.radians(60.0), W / H, 0.1, 100.0)
    prev_vp = proj * camera.view_matrix()

    check_frames = [0, 10, 50, 100]
    results = {}

    for frame_idx in range(max(check_frames) + 1):
        dt = 1.0 / 60.0
        cube.update(dt)
        vp = proj * camera.view_matrix()
        model = cube.model_matrix()

        renderer.render_scene(model, vp, cube.prev_model, prev_vp)
        renderer.warp_noise()
        renderer.display(mode=0)

        prev_vp = vp
        pygame.display.flip()

        if frame_idx in check_frames:
            noise = renderer.read_noise()
            results[frame_idx] = validate_noise(noise, frame_idx)

    pygame.quit()

    # Summary
    all_pass = all(
        abs(r["mean"]) < 0.1
        and 0.8 < r["std"] < 1.2
        and not r["has_nan"]
        and not r["has_inf"]
        for r in results.values()
    )
    print("\nOverall: %s" % ("ALL PASS" if all_pass else "SOME FAILED"))
    return results


if __name__ == "__main__":
    fire.Fire(run_validation)
