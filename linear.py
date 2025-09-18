import torch
import matplotlib.pyplot as plt

def produce_X(x):
    x0 =torch.ones(x.numpy().size)
    X = torch.stack((x,x0),dim=1)
    return X

x = torch.tensor([1.4,5,11,16,21])
y = torch.tensor([14.4,29.6,62,85.5,113.4])
X = produce_X(x)

inputs = X
targets = y

w = torch.rand(2,requires_grad=True)

def train(epochs=1,learning_rate=0.001):
    for epoch in range(epochs):
        output = inputs.mv(w)
        loss = (output-targets).pow(2).sum()

        loss.backward()
        w.data = (w.data - learning_rate*w.grad)
        w.grad.zero_()

        if epoch%80==0:
            draw(output,loss)
    return w,loss

def draw(output,loss):
    plt.cla()
    plt.scatter(x.numpy(),y.numpy())
    plt.plot(x.numpy(),output.data.numpy(),'r-',lw=5)
    plt.text(0.5,0,'Loss: {:.4f}'.format(loss),fontdict={'size':20,'color':'red'})
    plt.pause(0.005)

w,loss = train(10000,learning_rate=0.001)

print("Loss: {:.4f}".format(loss))
print("W: {:.4f}",w.data)
