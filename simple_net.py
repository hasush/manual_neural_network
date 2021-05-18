import numpy as np
import matplotlib.pyplot as plt 

class Network(object):
    """ https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ """

    def __init__(self):
        self.w1=0.15
        self.w2=0.20
        self.w3=0.25
        self.w4=0.30
        self.w5=0.40
        self.w6=0.45
        self.w7=0.50
        self.w8=0.55
        self.b1=0.35
        self.b2=0.60
        self.input1=0.05
        self.input2=0.10
        self.target1=0.01
        self.target2=0.99
        self.h1=None
        self.h2=None
        self.lr=0.5
        self.num_iterations=10000
    
    def get_targets(self):
        return self.target1,self.target2

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def forward(self):
        self.h1=self.input1*self.w1 + self.input2*self.w2 + self.b1
        self.a1=self.sigmoid(self.h1)
        self.h2=self.input1*self.w3 + self.input2*self.w4 + self.b1
        self.a2=self.sigmoid(self.h2)
        self.output1=self.a1*self.w5 + self.a2*self.w6 + self.b2
        self.a3=self.sigmoid(self.output1)
        self.output2=self.a1*self.w7 + self.a2*self.w8 + self.b2
        self.a4=self.sigmoid(self.output2)
        return self.a3, self.a4

    def compute_loss(self, targets, outputs):
        loss=0
        for target,output in zip(targets,outputs):
            loss+=0.5*(target-output)**2
        return loss

    def back_propagate(self):
        
        # Compute derivatives.
        dloss1a3=(self.target1-self.a3)*-1
        dloss2a4=(self.target2-self.a4)*-1

        da3do1=(1-self.a3)*self.a3
        da4do2=(1-self.a4)*self.a4

        do1dw5=self.a1 
        do1dw6=self.a2 
        do2dw7=self.a1 
        do2dw8=self.a2 

        do1da1=self.w5
        do1da2=self.w6
        do2da1=self.w7
        do2da2=self.w8

        da1dh1=(1-self.a1)*self.a1 
        da2dh2=(1-self.a2)*self.a2

        dh1dw1=self.input1 
        dh1dw2=self.input2 
        dh2dw3=self.input1 
        dh2dw4=self.input2 

        dloss1dw5=dloss1a3*da3do1*do1dw5
        dloss1dw6=dloss1a3*da3do1*do1dw6
        dloss2dw7=dloss2a4*da4do2*do2dw7
        dloss2dw8=dloss2a4*da4do2*do2dw8

        dlossdw1=(dloss1a3*da3do1*do1da1 + dloss2a4*da4do2*do2da1)*da1dh1*dh1dw1
        dlossdw2=(dloss1a3*da3do1*do1da1 + dloss2a4*da4do2*do2da1)*da1dh1*dh1dw2
        dlossdw3=(dloss1a3*da3do1*do1da2 + dloss2a4*da4do2*do2da2)*da2dh2*dh2dw3
        dlossdw4=(dloss1a3*da3do1*do1da2 + dloss2a4*da4do2*do2da2)*da2dh2*dh2dw4

        # Update weights. 
        self.w1-=self.lr*dlossdw1
        self.w2-=self.lr*dlossdw2
        self.w3-=self.lr*dlossdw3
        self.w4-=self.lr*dlossdw4
        self.w5-=self.lr*dloss1dw5
        self.w6-=self.lr*dloss1dw6
        self.w7-=self.lr*dloss2dw7
        self.w8-=self.lr*dloss2dw8
        self.b2-=self.lr*(dloss1a3*da3do1 + dloss2a4*da4do2)
        self.b1-=self.lr*(dloss1a3*da3do1*(do1da1*da1dh1 + do1da2*da2dh2) + dloss2a4*da4do2*(do2da1*da1dh1+do2da2*da2dh2))

    def sgd(self):

        losses=[]
        for iteration in range(self.num_iterations):
            outputs=self.forward()
            targets=self.get_targets()
            loss=self.compute_loss(targets, outputs)
            losses.append(loss)
            if (iteration+1)%500==0:
                print(f"Loss: {loss} for iteration: {iteration} -- targets/outputs: {targets}/{outputs}")
            self.back_propagate() 
        plt.figure()
        plt.plot(losses)
        plt.show()       

def main():
    network=Network()
    network.sgd()

if __name__ == "__main__":
    main()