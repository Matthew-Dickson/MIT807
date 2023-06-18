import torch

'''Return the device that is available to be used for model training '''
def get_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")


'''Sends model to be run on device specified'''
def to_device(object, device):
    if isinstance(object, (list, tuple)):
        return [to_device(x, device) for x in object]
    return object.to(device, non_blocking=False)


class ToDeviceLoader:
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    device = get_device()
    print('device: {}'.format(device))
    
