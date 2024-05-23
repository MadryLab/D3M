# Data Debiasing with Datamodels (D3M): Improving Group Robustness via Dataset Selection


To run our code, first install our library by cloning this repository and
running:
```
cd d3m
pip install -e .
```

Then, you can use the following code to debias your dataset:

```
from d3m import D3M

# Any pytorch model
model = ...

# A list of model checkpoints, i.e. a list of "state_dict" objects
checkpoints = ...

# Pytorch dataloaders for the training and validation sets
train_dataloader = ...
val_dataloader = ...

# A list of group indices, i.e. a list of indices, where each element is
# the index of the group of the corresponding validation example
group_indices = ...

d3m = D3M(model,
          checkpoints,
          train_dataloader,
          val_dataloader,
          group_indices)

debiased_train_indices = d3m.debias()

# Use debiased_train_indices to create a new dataloader
debiased_train_dataloader = ...

# Train the model with the new dataloader
train(model, debiased_train_dataloader, ...)
...
```
