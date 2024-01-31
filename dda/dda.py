
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
               ):
        """

        """
        # Step 1: compute TRAK scores

        # Step 2: compute group alignment scores

        # Step 3: construct new training set
        debiased_train_inds = get_debiased_train_indices()
