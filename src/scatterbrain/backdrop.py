import fitsio
import numpy as np
from tqdm import tqdm
from .utils import get_star_mask
from .designmatrix import (
    radial_design_matrix,
    cartesian_design_matrix,
    spline_design_matrix,
    strap_design_matrix,
)

try:
    import cupy as cp

    def load_image_cupy(fname):
        return cp.asarray(load_image_numpy(fname))

    def load_image(fname):
        return load_image_cupy(fname)


except ImportError:
    import numpy as cp

    def load_image(fname):
        return load_image_numpy(fname)


def load_image_numpy(fname):
    image = np.asarray(fitsio.read(fname)[:2048, 45 : 45 + 2048])
    image[~np.isfinite(image)] = 1e-5
    image[image <= 0] = 1e-5
    return image


# try:
#     import cupy as cp
#     from cupy import sparse
# except ImportError:
#     import numpy as cp
#     from scipy import sparse


class BackDrop(object):
    def __init__(self, column=None, row=None, ccd=3, sigma_f=None):
        self.A1 = radial_design_matrix(
            column=column, row=row, ccd=ccd, sigma_f=sigma_f, prior_mu=2, prior_sigma=3
        ) + cartesian_design_matrix(
            column=column, row=row, ccd=ccd, sigma_f=sigma_f, prior_mu=2, prior_sigma=3
        )
        self.A2 = spline_design_matrix(
            column=column, row=row, ccd=ccd, sigma_f=sigma_f, prior_sigma=100
        ) + strap_design_matrix(
            column=column, row=row, ccd=ccd, sigma_f=sigma_f, prior_sigma=100
        )
        self.column = column
        self.row = row
        self.ccd = ccd
        self.weights_basic = []
        self.weights_full = []

    def update_sigma_f(self, sigma_f):
        self.A1.update_sigma_f(sigma_f)
        self.A2.update_sigma_f(sigma_f)

    def clean(self):
        self.weights_basic = []
        self.weights_full = []

    def __repr__(self):
        return f"BackDrop CCD:{self.ccd} ({len(self.weights_basic)} frames)"

    def _build_masks(self, frame):
        if frame.shape != (2048, 2048):
            raise ValueError("Pass a frame that is (2048, 2048)")
        star_mask = get_star_mask(frame)
        sigma_f = cp.ones((2048, 2048))
        sigma_f[~star_mask] = 1e5
        self.update_sigma_f(sigma_f)
        return

    def model(tdx):
        return self.A1.dot(self.w[tdx])

    def _fit_basic(self, flux):
        self.weights_basic.append(self.A1.fit_frame(cp.log10(flux)))

    def _fit_full(self, flux):
        self.weights_full.append(self.A2.fit_frame(flux))

    def _model_basic(self, tdx):
        return cp.power(10, self.A1.dot(self.weights_basic[tdx])).reshape(self.shape)

    def _model_full(self, tdx):
        return self.A2.dot(self.weights_full[tdx]).reshape(self.shape)

    def model(self, tdx):
        return self._model_basic(tdx) + self._model_full(tdx)

    def fit_frame(self, frame):
        self._fit_basic(frame)
        res = frame - self._model_basic(-1)
        self._fit_full(res)
        res = res - self._model_full(-1)

        # SOMETHING ABOUT JITTER
        return

    def fit_model(self, flux_cube, test_frame=0):
        if flux_cube.ndim != 3:
            raise ValueError("`flux_cube` must be 3D")
        self._build_masks(flux_cube[test_frame])
        for flux in tqdm(flux_cube, desc="Fitting Frames"):
            self.fit_frame(flux)

    @property
    def shape(self):
        if self.column is not None:
            return (self.row.shape[0], self.column.shape[0])
        return (2048, 2048)

    @property
    def jitter():
        return

    def save(self, outfile="backdrop_weights.npz"):
        cp.savez(outfile, cp.asarray(self.weights_basic), cp.asarray(self.weights_full))

    def load(self, infile="backdrop_weights.npz"):
        cpzfile = cp.load(infile)
        self.weights_basic = list(cpzfile["arr_0"])
        self.weights_full = cpzfile["arr_1"]
        if self.column is not None:
            self.weights_full = np.hstack(
                [
                    self.weights_full[:, :-2048],
                    self.weights_full[
                        :, self.weights_full.shape[1] - 2048 + self.column
                    ],
                ]
            )
        self.weights_full = list(self.weights_full)
        return self

    @property
    def weights(self):
        self.w1 = None
        self.w2 = None
        return

    def plot(self):
        return
