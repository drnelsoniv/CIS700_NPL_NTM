import numpy as np
import torch as tr

# This code adapted from in-class example of NTM provided by Dr. Garrett Katz.

class NTM:
    def __init__(self):
        self.num_addresses = 3
        self.pattern_size = 10
        self.hidden_size = 64
        self.M = (tr.randn(self.num_addresses, self.pattern_size) > 0).float()
        
        print("randomized M: ")
        print(self.M)
    
        self.controller = tr.nn.LSTM(
            input_size = self.pattern_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True) # [batch index, time index, neuron index]
        self.readout = tr.nn.Linear(self.hidden_size, self.num_addresses)
        
        print(self.loss())
        
        self.params = list(self.controller.parameters()) + list(self.readout.parameters())
        
        self.optimize(self.loss, self.params, num_iters=1000, learning_rate=0.01, verbose=1)
        
        print("r: ", self.r)
        print("target r (M[2]): ", self.M[2])
        
    def optimize(self, loss, params, num_iters, learning_rate, verbose):
        learning_curve = []
        for i in range(num_iters):
            l = loss()
            if verbose > 0 and i % 100 == 0: print(i, l.item())
            learning_curve.append(l.item())
            
            l.backward()
            
            for p in params:
                p.data -= learning_rate * p.grad
                p.grad *= 0
        return learning_curve
    
    def loss(self):
        w = tr.tensor([[1.,0.,0.]])
        h = tr.zeros(1,1,self.hidden_size)
        c = tr.zeros(1,1,self.hidden_size)
        for i in range(3):
            self.r = self.read(w, self.M)
            lstm_out, (h,c) = self.controller(self.r.reshape(1,1,self.pattern_size), (h,c))
            w = tr.softmax(self.readout(lstm_out), dim=2).reshape(1, self.num_addresses)
        
        err = ((self.r - self.M[2])**2).sum()
        return err
    
    # Read/write operations
    
    def read(self, w, M):
        r = tr.mm(w, M)
        return r
        
    def write(self, w, M_prev, e, a):
        # Write to memory.
        # This is accomplished by erasing, and then adding to the memory
        # matrix from the previous time step according to the attention
        # vector 'w' which selects the memory location(s) to modify, along
        # with erase vector (e) and add vector (a) which determine how to
        # modify the data at that memory location.
        
        #print("M_prev", M_prev)
        #print("(1 - tr.mm(w, e))", (1 - tr.mm(w, e)))
        
        M_after_erase = M_prev * (1 - tr.mm(w, e))
        
        #print("M_after_erase", M_after_erase)
        #print("tr.mm(w,a)", tr.mm(w, a))
        
        M_new = M_after_erase + tr.mm(w, a)
        return M_new
        
    # Memory location weight matrix functions
    
    def cosine_similarity(self, a, b):
        return tr.mm(a, b.transpose(0,1)) / (((a**2).sum())**0.5 * ((b**2).sum())**0.5)
        
    def content(self, beta, k, M):
        result = tr.exp(beta * self.cosine_similarity(k, M))
        return result / result.sum()
        
    def interpolation(self, g, w_c, w_prev):
        return g*w_c + (1 - g)*w_prev
        
    def shift(self, w_g, s):
        return tr.mm(w_g, s.transpose(0,1))
        
    def sharpen(self, w_s, gamma):
        result = w_s**gamma
        return w_s / w_s.sum()
        
    def test(self):
        print("test:")
        
#        w = tr.tensor([[0.,0.,1.]]).transpose(0,1)
#        M_prev = tr.ones(3,5).float()
#        e = tr.tensor([[1.,1.,0.,1.,1.]])
#        a = tr.tensor([[0.,1.,0.,0.,0.]])
#        print("w ", w)
#        print("M_prev ", M_prev)
#        print("e ", e)
#        print("a ", a)
#        M_new = self.write(w, M_prev, e, a)
#        print("M_new ", M_new)

#        k = tr.tensor([[1.,0.,1.,0.,1.]])
#        M = tr.tensor([[0.,1.,0.,1.,0.],
#            [0.,1.,1.,0.,1.],
#            [1.,0.,1.,0.,1.]])
#        print("k ", k)
#        print("M ", M)
#        cos = self.cosine_similarity(k, M)
#        print("cos ", cos)
#        w_c = self.content(1, k, M)
#        print("w_c (B=1) ", w_c)
    