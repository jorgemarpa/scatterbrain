#!/usr/bin/env python

import argparse
import glob
import os
import time

# import numpy as np

os.environ['USE_CUPY'] = str(True)
from backdrop import BackDrop

from cupy_numpy_imports import load_image, load_image_numpy
import cupy as cp
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true", help="use mpi")
    parser.add_argument("--max-frames", type=int, default=0, help="maximum number of frames to process")
    parser.add_argument("--frames-per-rank", type=int, default=1, help="number of frames per rank to use in each batch")
    args = parser.parse_args()

    time_start = time.time()
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

    # Initialize list of frames on rank 0
    if rank == 0:
        # fnames = glob.glob('/nobackupp12/chedges/tess/sector01/camera1/ccd1/*ffic.fits.gz')
        fnames = glob.glob('/nobackupp19/chedges/hackday/ccd1/*ffic.fits.gz')
        # fnames = glob.glob('/tmp/*ffic.fits.gz')
        # fnames = list(range(1234))
        
        # init backdrop in rank 0
        b = BackDrop()
        
        if args.max_frames > 0:
            fnames = fnames[:args.max_frames]
        # initialize back drop here?
    else:
        fnames = None

    # Broadcast filenames to all other ranks
    if comm is not None:
        fnames = comm.bcast(fnames, root=0)
    else:
        # nothing to do if we are not using mpi
        pass

    num_files = len(fnames)
    frames_per_rank = args.frames_per_rank
    batch_size = frames_per_rank*size
    # round up in case batch_size does not evenly divide num_files
    num_batches = (num_files + batch_size - 1) // batch_size
    # if rank == 0:
    #     print("num_files      ", num_files)
    #     print("frames_per_rank", frames_per_rank)
    #     print("batch_size     ", batch_size)
    #     print("num_batches    ", num_batches)
    # return

    # read and process files in batches
    for batch_index in range(num_batches):
        time_batch_start = time.time()
        batch_start = batch_index * batch_size
        batch_stop = min((batch_index + 1) * batch_size, num_files)
        batch_fnames = fnames[batch_start:batch_stop]

        frames = []

        if rank < len(batch_fnames):
            # Each rank reads a different filename
            for i, fname in enumerate(batch_fnames[rank::size]):
                print(f"{batch_index=} rank {rank} will read: {fname}")
                frame = load_image_numpy(fname)
                frames.append(frame)
        else:
            # if batch_size does not evenly divide num_files, the final
            # batch will have some ranks that do no have any files to read in
            pass

        # gather frames to rank 0
        if comm is not None:
            # frames on rank 0 is a list of lists
            frames = comm.gather(frames, root=0)
            # unpack/flatten list of lists
            if rank == 0:
                # flatten and stack
                frames = cp.stack(cp.array([item for sublist in frames for item in sublist]))
            else:
                # other ranks have nothing to do
                pass
        else:
            frames = cp.stack(frames)
    
        if rank == 0:
            # process frames on rank 0 / perform back drop here
            result = cp.average(frames, axis=(-2, -1))
            print(f"{batch_index=} {result}")
            
            # backdrop stuff
            b.fit_model(frames)
            b.save(outfile=f"backdrop_output/backdrop_weights_batch{batch_index:03}.npz")
            
            print("A      :", type(b.A1.A))
            print("weights:", type(b.weights_basic[0]))
        else:
            # other ranks have nothing to do
            pass

    time_end = time.time()
    if rank == 0:
        print(f"time total: {time_end-time_start:.4f} s")


if __name__ == "__main__":
    main()
