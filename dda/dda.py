"""
This module implements the Debiasing through Data Attribution (DDA) method.
"""

import torch
import numpy as np
from torch.nn import functional as F

from .attrib import get_trak_matrix


class DDA:
    """
    Debiasing through Data Attribution
    """

    def __init__(
        self,
        model,
        checkpoints,
        train_dataloader,
        val_dataloader,
        group_indices,
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
            group_indices:
                A list indicating the group each sample in the validation
                dataset belongs to.
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
                pytorch device
        """
        self.model = model
        self.checkpoints = checkpoints
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.group_indices = group_indices
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

    def get_group_losses(self, model, val_dl, group_indices) -> list:
        """Returns a list of losses for each group in the validation set."""
        losses = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dl:
                outputs = model(inputs.to(self.device))
                loss = F.cross_entropy(
                    outputs, labels.to(self.device), reduction="none"
                )
                losses.append(loss)
        losses = torch.cat(losses)

        n_groups = len(set(group_indices))
        group_losses = [losses[group_indices == i].mean() for i in range(n_groups)]
        return group_losses

    def compute_group_alignment_scores(self, trak_scores, group_indices, group_losses):
        """
        Computes group alignment scores (check Section 3.2 in our paper for
        details).

        Args:
            trak_scores:
                result of get_trak_matrix
            group_indices:
                a list of the form [group_index(x) for x in train_dataset]

        Returns:
            a list of group alignment scores for each training example
        """
        n_groups = len(set(group_indices))
        S = np.array(trak_scores)
        g = [
            group_losses[i].cpu().numpy() * S[:, np.array(group_indices) == i].mean(axis=1)
            for i in range(n_groups)
        ]
        g = np.stack(g)
        group_alignment_scores = g.mean(axis=0)
        return group_alignment_scores

    def get_debiased_train_indices(
        self, group_alignment_scores, use_heuristic=True, num_to_discard=None
    ):
        """
        If use_heuristic is True, training examples with negative score will be discarded,
        and the parameter num_to_discard will be ignored
        Otherwise, the num_to_discard training examples with lowest scores will be discarded.
        """
        if use_heuristic:
            return [i for i, score in enumerate(group_alignment_scores) if score >= 0]

        if num_to_discard is None:
            raise ValueError("num_to_discard must be specified if not using heuristic.")

        sorted_indices = sorted(
            range(len(group_alignment_scores)),
            key=lambda i: group_alignment_scores[i],
        )
        return sorted_indices[num_to_discard:]

    def debias(self, use_heuristic=True, num_to_discard=None):
        """
        Debiases the training process by constructing a new training set that
        excludes examples which harm worst-group accuracy.

        Args:
            use_heuristic:
                If True, examples with negative group alignment scores are
                discarded.  If False, the `num_to_discard` examples with the
                lowest scores are discarded.
            num_to_discard:
                The number of training examples to discard based on their group
                alignment scores.  This parameter is ignored if `use_heuristic`
                is True.

        Returns:
            debiased_train_inds (list):
                A list of indices for the training examples that should be
                included in the debiased training set.
        """

        # Step 2 (Step 1 is to compute TRAK scores):
        # compute group alignment scores
        group_losses = self.get_group_losses(
            model=self.model,
            val_dl=self.dataloaders["val"],
            group_indices=self.group_indices,
        )

        group_alignment_scores = self.compute_group_alignment_scores(
            self.trak_scores, self.group_indices, group_losses
        )

        # Step 3:
        # construct new training set
        debiased_train_inds = self.get_debiased_train_indices(
            group_alignment_scores,
            use_heuristic=use_heuristic,
            num_to_discard=num_to_discard,
        )

        return debiased_train_inds
