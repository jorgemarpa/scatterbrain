#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys
import time

log = logging.getLogger(__name__)
# TESS image size
N = 2048

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cupy",
    dest="cupy",
    action="store_true",
    default=False,
    help="use Cupy, default is numpy",
)
parser.add_argument("--mpi", action="store_true", default=False, help="use mpi")
parser.add_argument(
    "--max-frames", type=int, default=0, help="maximum number of frames to process"
)
parser.add_argument(
    "--frames-per-rank",
    type=int,
    default=1,
    help="number of frames per rank to use in each batch",
)
parser.add_argument(
    "--buffer-gather",
    action="store_true",
    default=False,
    help="use buffer objects to gather frames",
)
parser.add_argument("--verbose", action="store_true", default=False, help="print info")
parser.add_argument(
    "--in-dir",
    type=str,
    dest="in_dir",
    default="/nobackupp19/chedges/hackday/ccd1/",
    help="path to directory with input FITS files",
)
parser.add_argument(
    "--out-dir",
    type=str,
    dest="out_dir",
    default="./outputs/",
    help="path to directory to save output files",
)
args = parser.parse_args()

# set verbose level for logger
if args.verbose:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# main program
def main():

    # set env variable that will be read by the cupy/numpy importer
    os.environ["USE_CUPY"] = str(args.cupy)
    from scatterbrain import BackDrop

    # import image loader and Numpy/Cupy
    from scatterbrain.cupy_numpy_imports import load_image_numpy, np, xp

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

    # Initialize list of fits files on rank 0
    if rank == 0:
        log.info(f"Using {xp.__file__.split('/')[-2]}")
        fnames = glob.glob(f"{args.in_dir}/*ffic.fits.gz")
        if args.max_frames > 0:
            fnames = fnames[: args.max_frames]

        # init Backdrop object in rank 0
        b = BackDrop()
    else:
        fnames = None

    # Broadcast filenames to all other ranks when using mpi
    if comm is not None:
        fnames = comm.bcast(fnames, root=0)
    else:
        # nothing to do if we are not using mpi
        pass

    # get batch size
    num_files = len(fnames)
    frames_per_rank = args.frames_per_rank
    batch_size = frames_per_rank * size
    # round up in case batch_size does not evenly divide num_files
    num_batches = (num_files + batch_size - 1) // batch_size
    log.info(f"{num_batches} batches of size {batch_size}")

    # read and process files in batches
    for batch_index in range(num_batches):
        # finding batch start and end accordingly
        batch_start = batch_index * batch_size
        batch_stop = min((batch_index + 1) * batch_size, num_files)
        batch_fnames = fnames[batch_start:batch_stop]

        # if using buffer gather, we need to create buffer variables to load each
        # batch and to agregate
        if args.buffer_gather:
            sendbuf = np.zeros((frames_per_rank, N, N), dtype=np.float64)
            if rank == 0:
                recvbuf = np.empty([size, frames_per_rank, N, N], dtype=np.float64)
            else:
                # other ranks do not need a receive buffer
                recvbuf = None
        else:
            # if no buffer gather, we append new items into list and then gather
            frames = []

        if rank < len(batch_fnames):
            # Each rank reads a different filename
            for i, fname in enumerate(batch_fnames[rank::size]):
                log.info(
                    f"{batch_index=} rank {rank} will read: {os.path.basename(fname)}"
                )
                frame = load_image_numpy(fname)
                if args.buffer_gather:
                    sendbuf[i] = frame
                else:
                    frames.append(frame)
        else:
            # if batch_size does not evenly divide num_files, the final
            # batch will have some ranks that do no have any files to read in
            pass

        # this barrier is for timing measurement only, other timing statements
        # do not have this barrrier because they are performed after collective
        # operations or only relevant to rank 0
        if comm is not None:
            comm.barrier()

        # gather frames to rank 0
        if comm is not None:
            if args.buffer_gather:
                comm.Gather(sendbuf, recvbuf, root=0)
                if rank == 0:
                    # flatten batch dimension
                    frames = recvbuf.reshape(batch_size, N, N)
            else:
                # frames on rank 0 is a list of lists
                frames = comm.gather(frames, root=0)
                # unpack/flatten list of lists
                if rank == 0:
                    frames = np.stack([item for sublist in frames for item in sublist])
                else:
                    # other ranks have nothing to do
                    pass
        else:
            frames = np.stack(frames)

        if rank == 0:
            # send to GPU if asked
            if args.cupy:
                frames = xp.array(frames)
            log.info(f"frame array is of {type(frames)}")
            # result = xp.average(frames, axis=(-2, -1))
            # print(f"{batch_index=} {result}")

            # process frames on rank 0 / perform back drop here
            b.fit_model(frames)
            b.save(
                outfile=f"{args.out_dir}/" f"backdrop_weights_batch{batch_index:03}.npz"
            )
        else:
            # other ranks have nothing to do
            pass

    time_end = time.time()
    if rank == 0:
        log.info(f"time total: {time_end-time_start:.4f} s")


if __name__ == "__main__":
    main()
