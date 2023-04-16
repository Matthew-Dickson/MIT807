import torchvision.transforms as transforms

def trainingAugmentation():
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
        # transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=[-0.01, 0.01]),
        # transforms.RandomAffine(degrees=(-15, 15), translate=(0, 0.1), scale=(0.9, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    return transform


def testingAugmentation():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    return transform