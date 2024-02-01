import numpy as np
import torch
import torchvision
from dda import DDA


def get_dataloader(
    batch_size=128, num_workers=5, split="train", shuffle=False, augment=True
):
    if augment:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(0),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
                ),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)
                ),
            ]
        )

    is_train = split == "train"
    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar/", download=True, train=is_train, transform=transforms
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
    )

    return loader


model = torchvision.models.resnet18().cuda().eval()

#  put some random init weights as a placeholder
ckpts = [model.state_dict()]

train_loader = get_dataloader()
val_loader = get_dataloader(split="val")

#  random group allocations as a placeholder
group_inds = [np.random.choice(4) for i in range(len(val_loader.dataset))]

dda = DDA(model, ckpts, train_loader, val_loader, group_inds)

debiased_inds = dda.debias()
