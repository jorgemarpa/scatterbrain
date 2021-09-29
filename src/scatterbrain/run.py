import os
import matplotlib.pyplot as plt
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--cupy",
    dest="cupy",
    action="store_true",
    default=False,
    help="Use Cupy, default is Numpy.",
)
# this flag is to test how much faster is to load arrays into GPU natively.
# needs *npy files with TESS images.
parser.add_argument(
    "--load-npy",
    dest="load_npy",
    action="store_true",
    default=False,
    help="Load images from npy files.",
)
parser.add_argument(
    "--n-frames",
    dest="n_frames",
    type=int,
    default=10,
    help="Number of TESS frames.",
)
args = parser.parse_args()

# set env variable that will be read by the cupy/numpy importer
os.environ["USE_CUPY"] = str(args.cupy)
from backdrop import BackDrop

# import image loader
from cupy_numpy_imports import load_image

# import numpy or cupy to be used inside this script
if args.cupy:
    import cupy as xp
else:
    import numpy as xp


def main():
    nf = args.n_frames
    # These are available to anyone using the jupyter-lab
    if args.load_npy:
        fnames = glob("/nobackupp12/jimartin/gpu_hack_2021/data/cupy_image_flux_*.npy")
    else:
        fnames = glob("/nobackupp12/chedges/tess/sector01/camera1/ccd1/*ffic.fits.gz")
    fs = xp.zeros((nf, 2048, 2048), dtype=xp.float32)
    for tdx, fname in enumerate(tqdm(fnames[:nf], desc="Loading data into memory")):
        if args.load_npy:
            fs[tdx] = xp.load(fname)
        else:
            fs[tdx] = load_image(fname)

    b = BackDrop()
    b.fit_model(fs)
    b.save()

    print("A      :", type(b.A1.A))
    print("weights:", type(b.weights_basic[0]))


if __name__ == "__main__":
    main()
    print("Done!")
