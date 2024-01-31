
class DDA():
    """
    Debiasing through Data Attribution
    """
    def __init__():
        pass

    def get_group_losses(model, val_dl):
        pass

    def compute_group_alignment_scores(trak_scores, group_indices, group_losses):
        """
        Computes group alignment scores (check Section 3.2 in our paper for
        details).

        Args:
            trak_scores:
                result of get_attrib_matrix
            group_indices:
                a list of the form [group_index(x) for x in train_dataset]

        Returns:
            a list of group alignment scores for each training example
        """


    def get_debiased_train_indices(group_alignment_scores, use_heuristic=True, num_to_discard=None):
        """
        If use_heuristic is True, training examples with negative score will be discarded,
        and the parameter num_to_discard will be ignored
        Otherwise, the num_to_discard training examples with lowest scores will be discarded.
        """
        if use_heuristic:
            return [i for i, score in enumerate(group_alignment_scores) if score >= 0]
        else:
            if num_to_discard is None:
                raise ValueError("num_to_discard must be specified if not using heuristic.")

            sorted_indices = sorted(range(len(group_alignment_scores)),
                                    key=lambda i: group_alignment_scores[i])
            return sorted_indices[num_to_discard:]

    def debias(model,
               checkpoints,
               train_dataloader,
               val_dataloader,
               group_indices,
               train_set_size,
               val_set_size,
               use_heuristic=True,
               num_to_discard=None,
               ):
        """
        Debiases the training process by constructing a new training set that
        excludes examples which harm worst-group accuracy.

        Args:
            model:
                The model being trained.
            checkpoints:
                A list of paths to model checkpoints for computing TRAK scores.
            train_dataloader:
                DataLoader for the training data.
            val_dataloader:
                DataLoader for the validation data.
            group_indices:
                A list where the ith element is the group index of the ith
                example in the training set.
            train_set_size:
                The number of examples in the training set.
            val_set_size:
                The number of examples in the validation set.
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
        # Step 1: compute TRAK scores
        trak_scores = get_attrib_matrix(TODO)

        # Step 2: compute group alignment scores
        losses = [loss(model(x)) for (x, y) in val_dataloader]  # TODO
        group_losses = TODO

        group_alignment_scores = compute_group_alignment_scores(trak_scores,
                                                                group_indices,
                                                                group_losses)

        # Step 3: construct new training set
        debiased_train_inds = get_debiased_train_indices(group_alignment_scores,
                                                         use_heuristic=use_heuristic,
                                                         num_to_discard=num_to_discard)

        return debiased_train_inds