import theano
import numpy as np
from theano import function
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams as RS

X, Y = T.dmatrices(2)
B = T.dvector()
components, updates = theano.scan(lambda x,y,b : T.tanh(T.dot(x,y) + b), sequences=X, non_sequences=[Y,B])

ele_comp = function([X,Y,B], components)

dim = 10
X_realization = np.ones((dim, dim), dtype='float64')
Y_realization = np.ones((dim, dim), dtype='float64')
prng = RS(seed=9000)
B_real = prng.normal((dim, ), avg=0, std=2, dtype='float64')
B_realization = function([], B_real)
print ele_comp(X_realization, Y_realization, B_realization())

###################################################################################################################
###################################Evaluating a polynomial#########################################################
###################################################################################################################
co_eff = T.dvector()
free_var = T.dscalar()
max_coeff = T.iscalar()

components, updates = theano.scan(lambda ce, power, fv: ce*(fv**power), sequences=[co_eff, T.arange(max_coeff)], outputs_info=None,
                                  non_sequences=free_var) #outputs_info=None
poly_eval = components.sum()

polynomial_eval = function([co_eff, free_var, max_coeff], poly_eval)

maxc_real = 100
ce_real = prng.uniform(size=(maxc_real,), low=-100, high=100, dtype='float64')
ce_realf = function([], ce_real)
free_real = .89

print "Evaluation is:", polynomial_eval(ce_realf(), free_real, maxc_real)