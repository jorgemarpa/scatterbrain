"""basic class?"""
from abc import ABC, abstractmethod
from copy import deepcopy

from .cupy_numpy_imports import sparse, xp
from .utils import _spline_basis_vector


class design_matrix(ABC):
    def __init__(self, name="DM", sigma_f=None, prior_sigma=None, prior_mu=None):
        self.name = name
        self.A = self._build()
        if prior_mu is None:
            self.prior_mu = xp.zeros(self.shape[1])
        else:
            self.prior_mu = prior_mu
        if prior_sigma is None:
            self.prior_sigma = xp.ones(self.shape[1]) * xp.inf
        else:
            self.prior_sigma = prior_sigma
        self._validate()
        self.update_sigma_f(sigma_f)

    def update_sigma_f(self, sigma_f):
        if sigma_f is None:
            self.sigma_f = xp.ones(self.shape[0])
        elif sigma_f.ndim == 2:
            self.sigma_f = sigma_f.ravel()
        else:
            self.sigma_f = sigma_f
        if self.A is None:
            return
        if not self.sigma_f.shape[0] == self.shape[0]:
            raise ValueError(f"`sigma_f` must be shape {self.shape[0]}")
        self._build_sigma_w_inv()

    def _validate(self):
        if self.A is None:
            return
        if isinstance(self.prior_mu, (int, float)):
            self.prior_mu = xp.ones(self.shape[1]) * self.prior_mu
        else:
            if not self.prior_mu.shape[0] == self.shape[1]:
                raise ValueError(f"`prior_mu` must be shape {self.shape[1]}")
        if isinstance(self.prior_sigma, (int, float)):
            self.prior_sigma = xp.ones(self.shape[1]) * self.prior_sigma
        else:
            if not self.prior_sigma.shape[0] == self.shape[1]:
                raise ValueError(f"`prior_sigma` must be shape {self.shape[1]}")

    def _build_sigma_w_inv(self):
        self.AT = self.A.T
        if sparse.issparse(self.A):
            self.sigma_w_inv = self.AT.dot(
                self.A.multiply(sparse.csr_matrix(1 / self.sigma_f[:, None]))
            ) + xp.diag(1 / self.prior_sigma ** 2)
        else:
            self.sigma_w_inv = self.AT.dot(self.A / self.sigma_f[:, None]) + xp.diag(
                1 / self.prior_sigma ** 2
            )

    @abstractmethod
    def _build(self):
        return

    @property
    def shape(self):
        return self.A.shape

    def __repr__(self):
        return f"{self.name} {self.shape}"

    def copy(self):
        return deepcopy(self)

    def join(self, other):
        copy = self.copy()
        if sparse.issparse(copy.A) and sparse.issparse(other.A):
            copy.A = sparse.hstack([copy.A, other.A]).tocsr()
        elif (not sparse.issparse(copy.A)) and (not sparse.issparse(other.A)):
            copy.A = xp.hstack([copy.A, other.A])
        elif sparse.issparse(copy.A) and (not sparse.issparse(other.A)):
            copy.A = sparse.hstack(
                [copy.A, sparse.csr_matrix(deepcopy(other.A))]
            ).tocsr()
        else:
            copy.A = sparse.hstack([sparse.csr_matrix(copy.A), other.A]).tocsr()
        copy.prior_mu = xp.hstack([copy.prior_mu, other.prior_mu])
        copy.prior_sigma = xp.hstack([copy.prior_sigma, other.prior_sigma])
        copy.sigma_f = xp.hypot(copy.sigma_f, other.sigma_f)
        copy._build_sigma_w_inv()
        copy.name = " and ".join([copy.name, other.name])
        return copy

    def __add__(self, other):
        return self.join(other)

    def fit_frame(self, flux):
        B = (
            self.AT.dot(flux.ravel() / self.sigma_f)
            + self.prior_mu / self.prior_sigma ** 2
        )
        return xp.linalg.solve(self.sigma_w_inv, B)

    def fit_batch(self, flux_cube):
        B = xp.asarray(
            [
                self.AT.dot(flux.ravel() / self.sigma_f)
                + self.prior_mu / self.prior_sigma ** 2
                for flux in flux_cube
            ]
        ).T
        return xp.linalg.solve(self.sigma_w_inv, B).T

    def dot(self, other):
        return self.A.dot(other)


class TESS_design_matrix(design_matrix):
    def __build(self):
        raise ValueError("Can not instantiate a `TESS_design_matrix`")

    def __init__(
        self,
        sigma_f=None,
        prior_sigma=None,
        prior_mu=None,
        ccd=3,
        column=None,
        row=None,
        name="TESS",
        cutout_size=2048,
    ):
        self.cutout_size = cutout_size
        self.ccd = ccd
        if self.ccd in [1, 3]:
            self.bore_pixel = [2048, 2048]
        elif self.ccd in [2, 4]:
            self.bore_pixel = [2048, 0]
        if (column is None) and (row is None):
            row, column = xp.mgrid[: self.cutout_size, : self.cutout_size]
            self.column, self.row = (column - self.bore_pixel[1]) / (2048), (
                row - self.bore_pixel[0]
            ) / (2048)
        elif (column is not None) and (row is not None):
            #             row, column = xp.meshgrid(column, row)

            row, column = xp.meshgrid(row, column)
            row = row.T
            column = column.T
            self.column, self.row = (column - self.bore_pixel[1]) / (
                self.cutout_size
            ), (row - self.bore_pixel[0]) / (self.cutout_size)
        else:
            raise ValueError("Specify both column and row")
        super().__init__(
            name=name, sigma_f=sigma_f, prior_sigma=prior_sigma, prior_mu=prior_mu
        )
        self._validate()


class cartesian_design_matrix(TESS_design_matrix):
    def _build(self):
        A1 = xp.vstack([self.column.ravel() ** idx for idx in range(self.npoly)]).T
        A2 = xp.vstack([self.row.ravel() ** idx for idx in range(self.npoly)]).T
        return xp.hstack(
            [A1 * A2[:, idx][:, None] for idx in xp.arange(0, A2.shape[1])]
        )

    def __init__(
        self,
        sigma_f=None,
        prior_sigma=None,
        prior_mu=None,
        ccd=3,
        npoly=5,
        column=None,
        row=None,
        cutout_size=2048,
    ):
        self.npoly = npoly
        super().__init__(
            name="cartesian",
            sigma_f=sigma_f,
            prior_sigma=prior_sigma,
            prior_mu=prior_mu,
            ccd=ccd,
            column=column,
            row=row,
            cutout_size=cutout_size,
        )


class radial_design_matrix(TESS_design_matrix):
    def _build(self):
        self.rad = xp.hypot(self.column, self.row).ravel() / xp.sqrt(2)
        A = xp.vstack([self.rad.ravel() ** idx for idx in range(self.npoly)]).T
        return A

    def __init__(
        self,
        sigma_f=None,
        prior_sigma=None,
        prior_mu=None,
        ccd=3,
        npoly=10,
        column=None,
        row=None,
        cutout_size=2048,
    ):
        self.npoly = npoly
        super().__init__(
            name="radial",
            sigma_f=sigma_f,
            prior_sigma=prior_sigma,
            prior_mu=prior_mu,
            ccd=ccd,
            column=column,
            row=row,
            cutout_size=cutout_size,
        )


class strap_design_matrix(TESS_design_matrix):
    def _build(self):
        d = sparse.csr_matrix(xp.diag(xp.ones(self.column.shape[1])))
        return sparse.hstack([d] * self.column.shape[0]).T.tocsr()

    def __init__(
        self,
        sigma_f=None,
        prior_sigma=None,
        prior_mu=None,
        ccd=3,
        npoly=10,
        column=None,
        row=None,
        cutout_size=2048,
    ):
        self.npoly = npoly
        super().__init__(
            name="strap",
            sigma_f=sigma_f,
            prior_sigma=prior_sigma,
            prior_mu=prior_mu,
            ccd=ccd,
            column=column,
            row=row,
            cutout_size=cutout_size,
        )


class spline_design_matrix(TESS_design_matrix):
    def _build(self):
        """Builds a 2048**2 x N matrix"""
        x = self.column[0] + (self.bore_pixel[1] / self.cutout_size)
        knots = (
            xp.linspace(0, 1, self.nknots) + 1e-10
        )  # This stops numerical instabilities where x==knot value
        knots_wbounds = xp.append(
            xp.append([0] * (self.degree - 1), knots), [1] * (self.degree + 2)
        )

        # 2D sparse matrix, for 2048 pixels
        As = sparse.vstack(
            [
                sparse.csr_matrix(
                    _spline_basis_vector(x, self.degree, i, knots_wbounds)
                )
                for i in xp.arange(-1, len(knots_wbounds) - self.degree - 3)
            ]
        ).T

        # 2D sparse matrix, for 2048 pixels x 2048 columns
        A1 = sparse.vstack([As] * self.column.shape[0]).tocsr()

        # 2D sparse matrix, for 2048 pixels x 2048 rows
        if self.column.shape == (self.cutout_size, self.cutout_size):
            A2 = sparse.vstack(
                [A1[idx :: self.cutout_size] for idx in range(self.cutout_size)]
            ).tocsr()
        else:
            x = self.row[:, 0] + +(self.bore_pixel[1] / self.cutout_size)
            As = sparse.vstack(
                [
                    sparse.csr_matrix(
                        _spline_basis_vector(x, self.degree, i, knots_wbounds)
                    )
                    for i in xp.arange(-1, len(knots_wbounds) - self.degree - 3)
                ]
            ).T
            A2 = sparse.vstack([As] * self.column.shape[1]).tocsr()
            A2 = sparse.vstack(
                [A2[idx :: self.column.shape[0]] for idx in range(self.column.shape[0])]
            ).tocsr()
        return sparse.hstack(
            [A1.multiply(A2[:, idx]) for idx in range(A2.shape[1])]
        ).tocsr()

    def __init__(
        self,
        sigma_f=None,
        prior_sigma=None,
        prior_mu=None,
        ccd=3,
        nknots=40,
        degree=2,
        column=None,
        row=None,
        cutout_size=2048,
    ):
        self.degree = degree
        self.nknots = nknots
        super().__init__(
            name="spline",
            sigma_f=sigma_f,
            prior_sigma=prior_sigma,
            prior_mu=prior_mu,
            ccd=ccd,
            column=column,
            row=row,
            cutout_size=cutout_size,
        )
