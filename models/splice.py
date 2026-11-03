"""
SpLiCE model from "SpLiCE: Sparse Concept Decomposition for Interpretable CLIP Models"
"""

import logging

import torch

from models.base import InterpretableCLIP

logger = logging.getLogger(__name__)


class SpLiCE(InterpretableCLIP):
    """SpLiCE model from 'SpLiCE: Sparse Concept Decomposition for Interpretable CLIP Models'."""

    def __init__(
        self,
        features_size: int,
        text_concepts: list[str],
        concepts_embds: torch.Tensor,
        solver: str = "admm",
        l1_penalty: float = 0.01,
    ):
        """SpLiCE model from 'SpLiCE: Sparse Concept Decomposition for Interpretable CLIP Models'.

        Parameters
        ----------
        features_size : int
            Input feature size
        text_concepts : List[str]
            Human-readable names of the concepts, in the same order as `concepts_embds`.
        concepts_embds : torch.Tensor
            Concepts embeddings. The visual mean to center on is set afterwards (`set_visual_mean`).
        solver : str, optional
            Solver of the sparse decomposition; only "admm" (the default) is supported
        l1_penalty : float, optional
            L1 regularization penalty, by default 0.01
        """
        super().__init__(features_size, text_concepts, concepts_embds)

        logger.info("Initializing SpLiCE model.")

        if solver != "admm":
            raise ValueError(f"Solver {solver} not supported, only 'admm'")
        self.solver = solver

        self.rho = 5
        self.tol = 1e-6
        self.admm = ADMM(rho=self.rho, l1_penalty=l1_penalty, tol=self.tol, max_iter=2000, verbose=False)

    def _decompose(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decompose a dense CLIP embedding into a sparse weight vector

        Parameters
        ----------
        embedding : torch.Tensor
            A {batch x CLIP dimensionality} vector or batch of vectors.

        Returns
        -------
        weights : torch.Tensor
            A {batch x num_concepts} sparse vector over concepts.
        """
        return self.admm.fit(self.concepts.weight, embedding)

    def _recompose_image(self, weights):
        """recompose Converts a set of weights into a reconstructed dense embedding

        Parameters
        ----------
        weights : torch.tensor
            A {batch x num_concepts} tensor of sparse weights.

        Returns
        -------
        recon_image : torch.tensor
            A {batch x CLIP dimensionality} tensor of dense reconstructions.
        """
        recon_image = weights @ self.concepts.weight
        recon_image = torch.nn.functional.normalize(recon_image, dim=1)
        recon_image = torch.nn.functional.normalize(recon_image + self.mean_visual_embds, dim=1)
        return recon_image

    def forward(self, visual_features: torch.Tensor, tags: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        """Encode the visual features (SpLiCE is training-free: no loss).

        Parameters
        ----------
        visual_features : torch.Tensor
            Visual features
        tags : Optional[torch.Tensor], optional
            Optional sample-level tags to be used for disentanglement losses, by default None

        Returns
        -------
        tuple[torch.Tensor, float, dict[str, torch.Tensor]]
            Encoded visual features, total loss (0) and loss components (none)
        """
        return self.encode_image(visual_features), 0.0, {}

    def encode_image(self, visual_features: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode visual features.

        Parameters
        ----------
        visual_features : torch.Tensor
            Input visual features.
        return_weights : bool, optional
            Whether to also return the reconstruction weights, by default False

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            The reconstructed visual features, or, if `return_weights` is True, a tuple of
            (reconstructed visual features, sparse concept weights).
        """

        visual_features = torch.nn.functional.normalize(visual_features, dim=1)
        centered_visual_features = torch.nn.functional.normalize(visual_features - self.mean_visual_embds, dim=1)

        weights = self._decompose(centered_visual_features)

        recon_image = self._recompose_image(weights)

        if return_weights:
            return recon_image, weights

        return recon_image


class ADMM:
    def __init__(self, rho=1.0, l1_penalty=0.2, tol=1e-6, max_iter=10000, verbose=False):
        self.rho = rho
        self.l1_penalty = l1_penalty
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def step(self, Cb, solve, z, u):
        xn = solve(2 * Cb + self.rho * (z - u))
        zn = torch.where((xn + u - self.l1_penalty / self.rho) > 0, xn + u - self.l1_penalty / self.rho, 0)
        un = u + xn - zn
        return xn, zn, un

    def fit(self, C, v):
        ## iterates are in c, the number of concepts
        c, d = C.shape
        device = C.device

        if c > d:
            # Woodbury identity: (2 C C^T + rho I_c)^-1 r = (r - C (C^T C + rho/2 I_d)^-1 C^T r) / rho.
            # Same solution, but factors a d x d matrix instead of c x c (e.g. 512 vs 15k LAION concepts).
            M_cho = torch.linalg.cholesky(C.T @ C + (torch.eye(d) * self.rho / 2).to(device))

            def solve(r):
                return (r - C @ torch.cholesky_solve(C.T @ r, M_cho)) / self.rho

        else:
            ## size: c x c
            Q = 2 * C @ C.T + (torch.eye(c) * self.rho).to(device)

            # factor Q for quicker solve -- this is critical.
            Q_cho = torch.linalg.cholesky(Q)

            def solve(r):
                return torch.cholesky_solve(r, Q_cho)

        # iterates, size: c x batch
        # drawn on the CPU generator and then moved, so the initialization doesn't depend on the device
        x = torch.randn((c, v.shape[0])).to(device)
        z = torch.randn((c, v.shape[0])).to(device)
        u = torch.randn((c, v.shape[0])).to(device)

        for ix in range(self.max_iter):  # noqa: B007 (logged after the loop)
            z_old = z
            x, z, u = self.step(C @ v.T, solve, z, u)

            res_prim = torch.linalg.norm(x - z, dim=0)
            res_dual = torch.linalg.norm(self.rho * (z - z_old), dim=0)

            if (res_prim.max() < self.tol) and (res_dual.max() < self.tol):
                break
        if self.verbose:
            logger.info("ADMM stopped at iteration %d (primal residual %.3g, dual residual %.3g)", ix, res_prim.mean(), res_dual.mean())
        return z.T
