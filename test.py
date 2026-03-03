import sys
import os
import platform
code_path = os.path.join(os.path.dirname(__file__), 'src')
# Add it to the system path
sys.path.append(os.path.abspath(code_path))
from warp_particle import *
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True, help='Name of experiment')
parser.add_argument('-m', '--mode', type=str, help='Mode', default="particle")
parser.add_argument('-f', '--frames', type=int, help='Max number of frames to process', default=None)
parser.add_argument('--cpu', action='store_true', help='Use CPU backend (slower but portable)')
args = parser.parse_args()

if args.cpu or platform.system() == 'Darwin':
    # macOS: Taichi Metal backend has unreliable float atomics and no sparse SNodes.
    # CPU with multithread is correct and fast on Apple Silicon.
    fp = ti.f64
    ti.init(arch=ti.cpu, debug=False, default_fp=fp, random_seed=0)
else:
    fp = ti.f64
    ti.init(arch=ti.gpu, device_memory_GB=4.0, debug=False, default_fp=fp, random_seed=0)

exp_name = args.name
exp_mode = args.mode
# load map
flow_map = np.load(os.path.join("data", exp_name, "flows.npy")) # num_frames, H, W, 2
if args.frames is not None:
    flow_map = flow_map[:args.frames]
print("Loaded flow map has shape: ", flow_map.shape)
n_frames, H, W, _ = flow_map.shape
n_noise_channels = 4
init_noise = np.random.randn(H, W, n_noise_channels)

logs_dir = os.path.join("logs", exp_name, exp_mode)
os.makedirs(logs_dir, exist_ok=True)

# assume input is [height, width, 3]
def save_sample(sample, results_dir, name, vmin = 0, vmax = 1):
    image_processed = np.clip(sample, vmin, vmax)
    vrange = vmax - vmin
    image_processed = (((image_processed  - vmin )/ vrange) * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_processed)
    image_pil.save(os.path.join(results_dir, name + ".jpg"))

# #
# identity maps (cell center)
i = np.arange(H) + 0.5
j = np.arange(W) + 0.5
ii, jj = np.meshgrid(i, j, indexing='ij')
identity_cc = np.stack((ii, jj), axis=-1)

warper = ParticleWarper(H, W, n_noise_channels, fp=fp)

results = [init_noise]
prev_noise = init_noise
for i, flow_map_i in enumerate(flow_map):
    print(f"Warping: {i}")
    warper.set_deformation(identity_cc - flow_map_i)
    warper.set_noise(prev_noise)
    warper.run()
    prev_noise = warper.noise_field.to_numpy()
    results.append(prev_noise)

results = np.array(results)
print("Saving .npy for warped noise...")
np.save(os.path.join(logs_dir, "warped_noise.npy"), results)
print("Done")
print("Saving .jpg for warped noise...")
for i, result in enumerate(results):
    print(f"Outputting: {i}")
    save_sample(result[...,:3], logs_dir, str(i), vmin = -2, vmax = 2)
print("Done")
