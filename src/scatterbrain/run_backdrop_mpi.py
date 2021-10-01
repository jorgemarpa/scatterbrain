#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys
import time

import cupy as cp
import numpy as np

log = logging.getLogger(__name__)

N = 2048


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cupy",
        dest="cupy",
        action="store_true",
        default=False,
        help="use Cupy, default is numpy",
    )
    parser.add_argument("--mpi", action="store_true", help="use mpi")
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
        help="use buffer objects to gather frames",
    )
    parser.add_argument("--verbose", action="store_true", help="print info")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # set env variable that will be read by the cupy/numpy importer
    os.environ["USE_CUPY"] = str(args.cupy)
    from backdrop import BackDrop
    # import image loader
    from cupy_numpy_imports import load_image, load_image_numpy

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
    time_mpi_init = time.time()
    if rank == 0:
        log.info(f"time mpi init: {time_mpi_init-time_start:.4f} s")

    # Initialize list of frames on rank 0
    if rank == 0:
        # fnames = glob.glob('/nobackupp12/chedges/tess/sector01/camera1/ccd1/*ffic.fits.gz')
        fnames = glob.glob("/nobackupp19/chedges/hackday/ccd1/*ffic.fits.gz")
        # fnames = list(range(1234))
        if args.max_frames > 0:
            fnames = fnames[: args.max_frames]

        # init backdrop in rank 0
        b = BackDrop()

    else:
        fnames = None
    time_fnames_init = time.time()
    if rank == 0:
        log.info(f"time fnames init: {time_fnames_init-time_mpi_init:.4f} s")

    # Broadcast filenames to all other ranks
    if comm is not None:
        fnames = comm.bcast(fnames, root=0)
    else:
        # nothing to do if we are not using mpi
        pass
    time_fnames_bcast = time.time()
    if rank == 0:
        log.info(f"time fnames bcast: {time_fnames_bcast-time_fnames_init:.4f} s")

    num_files = len(fnames)
    frames_per_rank = args.frames_per_rank
    batch_size = frames_per_rank * size
    # round up in case batch_size does not evenly divide num_files
    num_batches = (num_files + batch_size - 1) // batch_size

    # read and process files in batches
    for batch_index in range(num_batches):
        time_batch_start = time.time()
        batch_start = batch_index * batch_size
        batch_stop = min((batch_index + 1) * batch_size, num_files)
        batch_fnames = fnames[batch_start:batch_stop]

        if args.buffer_gather:
            sendbuf = np.zeros((frames_per_rank, N, N), dtype="i")
            if rank == 0:
                recvbuf = np.empty([size, frames_per_rank, N, N], dtype="i")
            else:
                # other ranks do not need a receive buffer
                recvbuf = None
        else:
            frames = []

        if rank < len(batch_fnames):
            # Each rank reads a different filename
            for i, fname in enumerate(batch_fnames[rank::size]):
                log.info(f"{batch_index=} rank {rank} will read: {fname}")
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
        # do not have this barrrier because they are performed after collective operations
        # or only relevant to rank 0
        if comm is not None:
            comm.barrier()
        time_batch_load = time.time()
        if rank == 0:
            log.info(f"time batch load: {time_batch_load-time_batch_start:.4f} s")

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

        time_batch_gather = time.time()
        if rank == 0:
            log.info(f"time batch gather: {time_batch_gather-time_batch_load:.4f} s")

        if rank == 0:
            # process frames on rank 0 / perform back drop here
            # result = np.average(frames, axis=(-2, -1))
            # log.info(f"{batch_index=} {result}")

            if args.cupy:
                frames = cp.array(frames)

            # do backdrop
            log.info(type(frames))
            b.fit_model(frames)
            b.save(
                outfile=f"backdrop_output/backdrop_weights_batch{batch_index:03}.npz"
            )

            log.info(f"A      : {type(b.A1.A)}")
            log.info(f"weights: {type(b.weights_basic[0])}")
        else:
            # other ranks have nothing to do
            pass

        time_batch_process = time.time()
        if rank == 0:
            log.info(
                f"time batch process: {time_batch_process-time_batch_gather:.4f} s"
            )

    time_end = time.time()
    if rank == 0:
        log.info(f"time total: {time_end-time_start:.4f} s")


if __name__ == "__main__":
    main()
