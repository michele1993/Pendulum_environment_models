import torch

a = torch.rand((10,3))
w = torch.rand(3, requires_grad=True)

y = torch.rand(10)

z = torch.matmul(a,w)



loss = sum((y-z)**2)




loss.backward()



with torch.no_grad():
    w -= 0.1 * w.grad

print(w.grad) # the gradient is still there, so need to set it to zero

print(w.grad_fn)

w.grad.zero_()

print(w.grad)

