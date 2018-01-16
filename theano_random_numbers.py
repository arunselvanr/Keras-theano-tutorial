from theano.tensor.shared_randomstreams import RandomStreams as RS
import numpy as np
from theano import function

srng = RS(seed=989) #The seed for the random number is set


#############Uniform random variables

rn_u = srng.uniform(size=(2,3), low=1, high=6) #this will generate a uniform random matrix of order 2x3.
print "Type of rn_u is ", rn_u.type
unif = function([], rn_u, no_default_updates=False) #If no_default_updates were set to True then we get the same random
print "First Uniform matrix ", unif()                                        #number all the time.
print "Second Uniform ", unif()

############Binomial random variables

rn_b = srng.binomial(size=(3,),n=100,p=.7) #Say we want to generate an array of 3 independent binomial rvs

binom = function([], rn_b, no_default_updates=True)
print "First Binomial vector ", binom()
print "Second Binomial without changing random number generator", binom()

############Normal RV

rn_n = srng.normal(size=(), avg=0.0, std=2.3)
norm = function([],rn_n)
print "Single Normal ", norm()

#############Random integer list

rn_i = srng.random_integers(size = (4, ), low=1, high=900)
inte = function([], rn_i)
print "Integer list ", inte()

#############Generating a permutation unifromly at random

rn_p = srng.permutation(size=(), n = 10)
perm = function([], rn_p)
print "Random permutation of 0 to 9", perm()

#############choosing from a list randomly

rn_list = srng.choice(size=(), a=[2,3, 4.5, 6], replace=True, p=[.5, 0, .5, 0], dtype='float64')
lis = function([], rn_list)
print "Choosing 3 times from the specified list ", lis()
print lis()
print lis()

rn_another_list = srng.choice(size=(), a=3, replace=True, p=None)
an_list = function([], rn_another_list)

print "Choosing 3 times from [0,1, 2] since a is scalar", an_list()
print an_list()
print an_list()


