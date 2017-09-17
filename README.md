

```python
import numpy as np
import numpy.random as rand
```


```python
# inputs and target outputs

X = np.asarray([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y = np.asarray([0.0,1.0,1.0,0.0])
```


```python
# layers

def relu(x):
    return np.maximum(0, x)

def sig(x):
    return 1.0/(1 + np.exp(-x))
```

# Forward Calculation

$\mathbf{z_1} = \mathbf{X}\mathbf{w_1} + b_1$

$\mathbf{h_1} = relu(\mathbf{z_1})$

$\mathbf{z_2} = \mathbf{X}\mathbf{w_2} + b_2$

$\mathbf{h_2} = relu(\mathbf{z_2})$

$\mathbf{H} = [h_1, h_2]$

$\mathbf{z_3} = \mathbf{H}\mathbf{w_3} + b_3$

$\mathbf{y'} = \sigma(z_3)$

# Grad Calculation

$L = -\frac{1}{n}[\mathbf{y}^\intercal\ln(\mathbf{y'}) + (1 - \mathbf{y})^\intercal\ln{(1 - \mathbf{y'})}]$

$\nabla_{w_3} L = (\frac{\partial z_3}{\partial w_3})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

$\nabla_{w_1} L = (\frac{\partial z_1}{\partial w_1})^\intercal(\frac{\partial h_1}{\partial z_1})^\intercal(\frac{\partial z_3}{\partial h_1})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

$\nabla_{w_2} L = (\frac{\partial z_2}{\partial w_2})^\intercal(\frac{\partial h_2}{\partial z_2})^\intercal(\frac{\partial z_3}{\partial h_2})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

$\frac{\partial L}{\partial b_1} = (\frac{\partial z_1}{\partial b_1})^\intercal(\frac{\partial h_1}{\partial z_1})^\intercal(\frac{\partial z_3}{\partial h_1})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

$\frac{\partial L}{\partial b_2} = (\frac{\partial z_2}{\partial b_2})^\intercal(\frac{\partial h_2}{\partial z_2})^\intercal(\frac{\partial z_3}{\partial h_2})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

$\frac{\partial L}{\partial b_3} = (\frac{\partial z_3}{\partial b_3})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

# Individual Grad

$\nabla_{y'} L = -\frac{1}{n}[\frac{\mathbf{y}}{\mathbf{y'}} - \frac{(1-\mathbf{y})}{(1-\mathbf{y'})}]$

$\frac{\partial y'}{\partial z_3} = diag(\mathbf{y'}\bigodot(1 - \mathbf{y'}))$

$\frac{\partial z_3}{\partial h_1} = \{\mathbf{w}_{3,1}\}^n$

$\frac{\partial z_3}{\partial h_2} = \{\mathbf{w}_{3,2}\}^n$

$\frac{\partial h_1}{\partial z_1} = diag(min(1, \mathbf{h_1}))$

$\frac{\partial h_2}{\partial z_2} = diag(min(1, \mathbf{h_2}))$

$\frac{\partial z_3}{\partial w_3} = \mathbf{H}$

$\frac{\partial z_1}{\partial w_1} = \mathbf{X}$

$\frac{\partial z_2}{\partial w_2} = \mathbf{X}$

$\frac{\partial z_1}{\partial b_1} = \frac{\partial z_2}{\partial b_2} = \frac{\partial z_3}{\partial b_3} = \mathbf{1}$

## Chain rule for vector case

$\nabla_{w_3} L = (\frac{\partial z_3}{\partial w_3})^\intercal(\frac{\partial y'}{\partial z_3})^\intercal\nabla_{y'} L$

A case like the previous one is calculated by the chain rule applied in reverse. As we want the result to be a column vector so we, effictively, transpose the gradient term, even if it is not explicitly annotated. Now, as the gradient is transposed, then the whole calculation needs to be transposed. The standard form of Jacobian is that the nominator is expanded in columns and denominator in rows. Now as the reverse is true for the transposed gradient, we need to transpose all the Jacobians, and reversing them of course, so that nominators and denominators are matched as the original chain rule.



```python
# define net
class Net:
    def __init__(self):
        
        # init params
        
        self.w1 = rand.uniform(low=-1.0/np.sqrt(2), high=1.0/np.sqrt(2),size=(2))
        self.w2 = rand.uniform(low=-1.0/np.sqrt(2), high=1.0/np.sqrt(2),size=(2))
        self.w3 = rand.uniform(low=-1.0/np.sqrt(2), high=1.0/np.sqrt(2),size=(2))
    
        self.b1 = rand.uniform()
        self.b2 = rand.uniform()
        self.b3 = rand.uniform()
        
    def forward(self, x):
        
        # forward propagation
        
        z = np.dot(x, self.w1) + self.b1
        self.h1 = relu(z)
        
        z = np.dot(x, self.w2) + self.b2
        self.h2 = relu(z)
        
        self.h = np.asarray([self.h1, self.h2]).transpose()
        
        z = np.dot(self.h, self.w3) + self.b3
        self.o = sig(z)
        
        return self.o
    
    def train(self, x, y):
        
        # do forward
        
        lr = 0.9
        
        self.forward(x)
        
        n = x.shape[0]
        
        loss = -(1.0/n)*(np.dot(y, np.log(self.o)) + np.dot(1 - y, np.log(1 - self.o)))
        
        # calculate grad
        
        l_o_grad = -(1.0/n)*np.diag(((y/self.o) - ((1 - y)/(1 - self.o))))
        o_z3_grad = np.diag(self.o * (1 - self.o))
        z3_h1_grad = np.diag(np.full((n,), self.w3[0]))
        z3_h2_grad = np.diag(np.full((n,), self.w3[1]))
        h1_z1_grad = np.diag(np.minimum(1, self.h1))
        h2_z2_grad = np.diag(np.minimum(1, self.h2))
        z1_w1_grad = np.copy(x)
        z2_w2_grad = np.copy(x)
        z3_w3_grad = np.copy(self.h)
        
        l_z1_grad = h1_z1_grad.transpose().dot(z3_h1_grad.transpose()).dot(o_z3_grad.transpose()).dot(l_o_grad.transpose())
        l_b1_grad = (1.0/n)*np.sum(np.ones((4,)).transpose().dot(l_z1_grad), axis=0)
        l_w1_grad = (1.0/n)*np.sum(z1_w1_grad.transpose().dot(l_z1_grad), axis=1)

        l_z2_grad = h2_z2_grad.transpose().dot(z3_h2_grad.transpose()).dot(o_z3_grad.transpose()).dot(l_o_grad.transpose())
        l_b2_grad = (1.0/n)*np.sum(np.ones((4,)).transpose().dot(l_z2_grad), axis=0)
        l_w2_grad = (1.0/n)*np.sum(z2_w2_grad.transpose().dot(l_z2_grad), axis=1)
        
        l_z3_grad = o_z3_grad.transpose().dot(l_o_grad.transpose())
        l_b3_grad = (1.0/n)*np.sum(np.ones((4,)).transpose().dot(l_z3_grad), axis=0)
        l_w3_grad = (1.0/n)*np.sum(z3_w3_grad.transpose().dot(l_z3_grad), axis=1)
        
        # update weights
        
        self.w1 = self.w1 - (lr * l_w1_grad)
        self.b1 = self.b1 - (lr * l_b1_grad)
        
        self.w2 = self.w2 - (lr * l_w2_grad)
        self.b2 = self.b2 - (lr * l_b2_grad)
        
        self.w3 = self.w3 - (lr * l_w3_grad)
        self.b3 = self.b3 - (lr * l_b3_grad)
        
        return loss
    
    def params(self):
        return (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)
```


```python
net = Net()

output = net.forward(X)

for i,x in enumerate(X):
    print("{}: {}".format(x, output[i]))

for i in range(1000):
    loss = net.train(X, y)
    
    if i % 10 == 0:
        print(loss)
        
output = net.forward(X)

for i,x in enumerate(X):
    print("{}: {}".format(x, output[i]))
    
```

    [ 0.  0.]: 0.664998496578
    [ 0.  1.]: 0.624808888412
    [ 1.  0.]: 0.667601856875
    [ 1.  1.]: 0.636321480932
    0.744869502657
    0.702058024645
    0.693644334337
    0.689672997975
    0.685760431266
    0.679985255619
    0.677611171674
    0.675966335924
    0.672659972483
    0.667100906468
    0.658185631099
    0.645869435936
    0.629711299855
    0.609088721388
    0.583463480496
    0.552614941759
    0.516839958402
    0.477058441141
    0.434754973099
    0.391740816312
    0.349811766717
    0.310435877187
    0.274579434537
    0.242694831925
    0.214820432336
    0.190720852545
    0.170015296239
    0.152271366339
    0.137063068631
    0.124001088355
    0.112744909067
    0.10300447711
    0.0945366003755
    0.0871391942147
    0.0806450695891
    0.0749160910536
    0.0698380424289
    0.0653162815532
    0.0612721432075
    0.057639999784
    0.0543648768529
    0.0514005257563
    0.0487078672957
    0.046253734269
    0.0440098536321
    0.0419520204672
    0.0400594254875
    0.0383141056049
    0.036700493349
    0.0352050459073
    0.0338159385028
    0.03252280994
    0.0313165506032
    0.0301891251337
    0.0291334235398
    0.0281431357109
    0.0272126452656
    0.0263369394353
    0.025511532293
    0.0247323991304
    0.0239959201817
    0.0232988322076
    0.0226381867162
    0.0220113138015
    0.0214157907555
    0.0208494147474
    0.0203101789789
    0.0197962518193
    0.0193059585018
    0.018837765027
    0.0183902639754
    0.0179621619721
    0.0175522685884
    0.0171594864937
    0.0167828026982
    0.0164212807519
    0.0160740537792
    0.0157403182501
    0.0154193283989
    0.0151103912141
    0.0148128619337
    0.0145261399877
    0.0142496653384
    0.0139829151725
    0.0137254009084
    0.0134766654842
    0.013236280895
    0.013003845956
    0.0127789842657
    0.01256134235
    0.012350587968
    0.0121464085654
    0.0119485098579
    0.011756614535
    0.0115704610706
    0.0113898026316
    0.0112144060745
    0.0110440510222
    0.0108785290142
    0.0107176427219
    [ 0.  0.]: 0.0168285016467
    [ 0.  1.]: 0.99548172724
    [ 1.  0.]: 0.995456270033
    [ 1.  1.]: 0.0160601540939



```python

```
