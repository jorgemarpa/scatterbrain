import os
import numpy as np
import fitsio

try:
    if os.getenv("USE_CUPY") in ["True", "T", "true"]:
        print("Usign Cupy")
        import cupy as xp
        from cupy import sparse
        from cupyx.lapack import posv as cholesky_solve

        def load_image_cupy(fname):
            return xp.asarray(load_image_numpy(fname))

        @TimeRangeDecorator()
        def load_image(fname):
            return load_image_cupy(fname)

    else:
        raise ImportError

except ImportError:
    print("Usign Numpy")
    import numpy as xp
    from scipy import sparse

    cholesky_solve = np.linalg.solve

    def load_image(fname):
        return load_image_numpy(fname)


def load_image_numpy(fname):
    image = np.asarray(fitsio.read(fname)[:2048, 45 : 45 + 2048])
    image[~np.isfinite(image)] = 1e-5
    image[image <= 0] = 1e-5
    return image
