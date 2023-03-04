import numpy as np
import pickle

model=pickle.load(open('myfirstnn.pickle','rb'))
class MyfirstNN(object):
    def __init__(self):
        self.weight=model['weight']
        self.bias=model['bias']

    def sigmoid_function(self,x_input):
        return 1/1+np.exp(-x_input)
    
    def neuralnet(self,x_input):
        result=np.dot(x_input,self.weight)+self.bias
        res=self.sigmoid_function(result)
        return res
    def predict(self,x_input):
        x_input=np.array(x_input)
        pred=self.neuralnet(x_input)
        pred=list(pred)[0]
        if pred:
            pred=round(pred)
        else:
            pred=0
        return pred
#x=MyfirstNN()
#input=[0,0]
#y=x.predict(input)
#print(y)

