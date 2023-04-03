import torch

def softmax_with_temperature(logits, temperature) -> torch.float32:
    return torch.exp(logits/temperature) / torch.sum(torch.exp(logits/temperature))

if __name__ == '__main__':
    logits = torch.tensor([1., 2., 3.])
    print('logits value: {} logits exponetialvalue type: {}'.format(logits,logits.dtype))
    logits_exp = torch.exp(logits)
    print('logits exponetial value: {} logits exponetial value type: {}'.format(logits_exp, logits_exp.dtype))

    #Larger values of temperature leads to a smoother distibution 
    TEMPERATURES = [1.,5.,7.,10.,100., 10000]
    for temperature in TEMPERATURES:
        result = softmax_with_temperature(logits, temperature)
        print('softmax value: {} softmax value type: {} temperature: {}'.format(result, result.dtype, temperature))

    
