"""
This module implements the AutoDDA method. AutoDDA does *not* require any group
label information (neither for the training nor the validation dataset!).
Instead, AutoDDA levereages the attribution (TRAK) scores to *automatically
discover* coherent groups in the validation dataset where the model struggles.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA

from .d3m import D3M
from .attrib import get_trak_matrix


class AutoDDA(D3M):
    def __init__(
        self,
        model,
        checkpoints,
        train_dataloader,
        val_dataloader,
        train_set_size=None,
        val_set_size=None,
        trak_scores=None,
        trak_kwargs=None,
        device="cuda",
    ) -> None:
        """
        Args:
            model:
                The model to be debiased.
            checkpoints:
                A list of model checkpoints (state dictionaries) for debiasing
                (used to compute TRAK scores).
            train_dataloader:
                DataLoader for the training dataset.
            val_dataloader:
                DataLoader for the validation dataset.
            train_set_size (optional):
                The size of the training dataset. Required if the dataloader
                does not have a dataset attribute.
            val_set_size (optional):
                The size of the validation dataset. Required if the dataloader
                does not have a dataset attribute.
            trak_scores (optional):
                Precomputed TRAK scores. If not provided, they will be computed
                from scratch.
            trak_kwargs (optional):
                Additional keyword arguments to be passed to
                `attrib.get_trak_matrix`.
            device (optional):
                torch device
        """
        self.model = model
        self.checkpoints = checkpoints
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.device = device

        if trak_scores is not None:
            self.trak_scores = trak_scores
        else:
            try:
                self.train_set_size = len(train_dataloader.dataset)
                self.val_set_size = len(val_dataloader.dataset)
            except AttributeError as e:
                print(
                    f"No dataset attribute found in train_dataloader or val_dataloader. {e}"
                )
                if train_set_size is None or val_set_size is None:
                    raise ValueError(
                        "train_set_size and val_set_size must be specified if "
                        "train_dataloader and val_dataloader do not have a "
                        "dataset attribute."
                    ) from e
                self.train_set_size = train_set_size
                self.val_set_size = val_set_size

            # Step 1: compute TRAK scores
            if trak_kwargs is not None:
                trak_scores = get_trak_matrix(
                    train_dl=self.dataloaders["train"],
                    val_dl=self.dataloaders["val"],
                    model=self.model,
                    ckpts=self.checkpoints,
                    train_set_size=self.train_set_size,
                    val_set_size=self.val_set_size,
                    **trak_kwargs,
                )
            else:
                trak_scores = get_trak_matrix(
                    train_dl=self.dataloaders["train"],
                    val_dl=self.dataloaders["val"],
                    model=self.model,
                    ckpts=self.checkpoints,
                    train_set_size=self.train_set_size,
                    val_set_size=self.val_set_size,
                )

            self.trak_scores = trak_scores

        val_labels = []
        for batch in self.dataloaders["val"]:
            val_labels.extend(batch[1].tolist())

        self.group_indices = self.get_pseudogroups(
            trak_scores=self.trak_scores,
            val_labels=val_labels,
        )

    def get_pseudogroups(self, trak_scores, val_labels):
        """
        Automatically discover coherent groups in the validation dataset using
        PCA on the TRAK scores.

        Args:
            trak_scores:
                A tensor of TRAK scores for the validation dataset.
            val_labels:
                A list of labels for the validation dataset.
        Returns:
            A list of pseudogroup indices.
        """

        # Normalize TRAK scores, and move to GPU
        S = torch.tensor(trak_scores.T).contiguous().cuda().float()
        S /= S.norm(dim=1, keepdim=True) + 1e-5

        # Group scores by label
        val_labels = np.array(val_labels)
        trak_scores_by_label = {}
        for label in set(val_labels):
            trak_scores_by_label[label] = S[val_labels == label]

        # Perform PCA on each TRAK matrix
        pseudogroups = np.zeros(self.val_set_size)

        pca_projs = {}
        for label, trak_matrix in trak_scores_by_label.items():
            pca = PCA(n_components=1)
            pca.fit(trak_matrix.cpu().numpy())
            pcs = torch.tensor(pca.components_).float().cuda()
            projs = (trak_matrix @ pcs.T).cpu().numpy().T
            pca_projs[label] = projs[0]

        for label, pca_cmp in pca_projs.items():
            _pseudogroups = np.zeros_like(pca_cmp)
            _pseudogroups[pca_cmp > 0] = 1.0
            _pseudogroups += 2 * label
            pseudogroups[val_labels == label] = _pseudogroups

        return pseudogroups.tolist()
