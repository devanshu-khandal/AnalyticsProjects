# import pandas as pd
# import numpy as np
# print(2)
# print(3)

import sklearn 
# In[1]:
import numpy as np

class PredictionModel():
    def __init__(self, mvname, prodname):
        self.mvname = mvname
        self.prodname = mvname
        # self.data
        self.hello_world()
    def __repr__(self):
        return "basicThing"
        # pass
    def hello_world(self):
        x = 1+2 #print("hello_world")
        self.data = x
        return x

y = PredictionModel("Krish","RakeshRoshan")

y.prodname
# y.hello_world()
# y.data
y

# In[2]:

class Inheritance(PredictionModel):
    def __init__(self, mvname, prodname, actor):
        super().__init__(mvname,prodname)
        self.actor = actor
        self.hello_world()
    def __repr__(self):
        return "basicThing"

Inheritance("Krish","RakeshRoshan","Hrithik")
# Inheritance()


# In[3]:
class Base:
    def __init__(self):
        self.value = "Base"

    def output(self):
        print(self.value)

class Derived(Base):
    def __init__(self):
        super().__init__()
        self.value = "Derived"
        # self.show()

    def show(self):
        print(self.value)
        super().output()

d = Derived()
d.show()
x=Base()
x.output()

# d.base.value
# Output:
# Derived
# Base

# In[4]:

class MyList(list):
    def __init__(self, *args):
        super().__init__(*args)

    def sum(self):
        return sum(self)

my_list = MyList([1, 2, 3, 4])
print(my_list.sum())  # Output: 10 


# %%
