#!/usr/bin/env python

import argparse
import glob
import os
import numpy as np
import cupy as cp
import fitsio

from cupy_numpy_imports import load_image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true", help="use mpi")
    parser.add_argument("--max-frames", type=int, default=0, help="maximum number of frames to process")
    args = parser.parse_args()

    if args.mpi:
        # initialize mpi if requested
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        # otherewise, setup placeholder values
        comm = None
        rank = 0
        size = 1

    print(f"Hello from rank {rank} of {size}")

    # Initialize list of frames on rank 0
    if rank == 0:
        fnames = glob.glob('/nobackupp12/chedges/tess/sector01/camera1/ccd1/*ffic.fits.gz')
        if args.max_frames > 0:
            fnames = fnames[:args.max_frames]
    else:
        fnames = None
    
    # Broadcast filenames to all other ranks
    if comm is not None:
        fnames = comm.bcast(fnames, root=0)

    # fs = np.zeros((args.max_frames // size, 2048, 2048), dtype=np.float32)
    
    # Each rank reads a different filename
    # iterate over files
    flux_list = []
    for fname in fnames[rank::size]:
        # print(f"rank {rank} will read: {tdx} {os.path.basename(fname)}")
        # fs[tdx] = load_image(fname)
        flux_list.append(load_image(fname))
        
    if comm is not None:
        flux_list = comm.gather(flux_list, root=0)
        # print(type(flux_list))
        # print(len(flux_list))
        if rank == 0:
            flux_list = [item for sublist in flux_list for item in sublist]
        
    flux_list = np.array(flux_list)
            
    if rank == 0:
        print(type(flux_list))
        print(flux_list.shape)
    # print(fs.shape)


if __name__ == "__main__":
    main()