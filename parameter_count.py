from models.resnet import resnet110, resnet32

if __name__ == '__main__':
    model = resnet32()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

