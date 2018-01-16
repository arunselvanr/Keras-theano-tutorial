import numpy as np
import theano, time
from theano import tensor as T
from theano import function
from theano.ifelse import  ifelse

#tensor.gt (greater than), .ge (greater than or equal to)
#Similarly there are lt, le, eq, ne
#The evaluation of the above are all element-wise

#time.clock() can be used to get the time at any point in the operation.
# tic_1 = time.clock()
# Operation
#tic_2 = time.clock()
#tic_2 - tic_1 gives the time taken to run Operation.

a, b = T.dscalars(2)
x, y = T.dvectors(2)

z_switch = T.switch(T.le(a,b), T.mean(x), T.max(y))
z_ifelse = ifelse(T.gt(a,b), T.max(x), T.mean(y))

f_switch = function([a, b, x, y], z_switch, mode=theano.Mode(linker='vm'))
f_ifelse = function([a, b, x, y], z_ifelse, mode=theano.Mode(linker='vm'))

value1 = 2.3
value2 = 3.44

vector_1 = np.ones((4000,))
vector_2 = [2.3, 4.5, 5.6, 7.8, 9 , 10, 11, 12, 13, 14, 576, 456, 32467, 43598]

print f_switch(value1, value2, vector_1, vector_2)
print f_ifelse(value1, value2, vector_1, vector_2)