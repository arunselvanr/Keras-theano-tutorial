#A quick tutorial on how to define a function in python.
#def function_name(parameter_1, ..., parameter_n):
#    operation_1
#    .
#    .
#    .
#    operation_m
#
#    return [expression_1, ..., expression_r]
#End of a quick tutorial on how to define a function in python.
import numpy as np
import keras

class Complex:
    def __init__(self, rpart, ipart): #This is only optional.
        self.r = rpart
        self.i = ipart
#__init__ can be used to initialize a class_object
    def print_it(self):
        print self.r, '+i', self.i

x = Complex(2.3, 5.6)
y = Complex(3.4, 7.675)

x.print_it() #Calling a function that is defined within a class
y.print_it()

class Add_Complex:
    def __init__(self, a, b):
        self.r = a.r + b.r
        self.i = a.i + b.i
        self.sum = Complex(self.r, self.i)

sum_xy = Add_Complex(x, y)
print sum_xy.sum.print_it() #calling an object which is in turn a class.
print_sum_xy = sum_xy.sum.print_it #Note that we have stored an object function
#as a function accessible outside and we can call it, as is done below, to print
#the same thing.
print print_sum_xy()

#Implementing a double ended queue as a class.
class double_ended_q:
    def __init__(self):
        self.list = []
        print '1. The list of operation available are'
        print '2. append_right'
        print '3. append_left'
        print '4. pop_right'
        print '5. pop_left'
        print '6. count_instances'
        print '7. reverse_list'

    def append_left(self, x):
        self.list.insert(0, x)
#list.insert(i, x) inserts the element x into position i. Since we only allow for left and
#right insertions in double ended queues, we only let the position be 0 or len(list)
    def append_right(self, x):
        self.list.append(x)
    def pop_left(self):
        self.list.pop(0)
    def pop_right(self):
        self.list.pop(len(self.list)-1)
    def count_instances(self, x):
        self.list.count(x)
    def reverse_list(self):
        self.list.reverse()

deq = double_ended_q()

deq.append_left(6.89)
deq.append_right(4.56)
print deq.list
deq.pop_left()
print deq.list

#Lets implement a few string operations as classes.
class string:
    def __init__(self, strr):
        self.str = strr

    def string_count(self, sr):
        print self.str.count(sr, 0, len(self.str)-len(sr))
# the string sr is searched within str starting at 0 and ending at length of str - length of sr.
    def string_endswith(self, sr):
        print self.str.endswith(sr, 0, len(self.str)-len(sr))
#Answers does there exist a substring of str that ends with sr.
    def string_startswith(self, sr):
        print self.str.startswith(sr, 0, len(self.str)- len(sr))
#Answers does there exist a substring that starts with sr
    def string_find(self, sr, beg, end):
        print self.str.find(sr, beg, end)
#The above finds a substring, but only between indices beg and end.
    def string_replace(self, old, new, count):
        self.srt = self.str.replace(old, new, count )
        print 'New String:', self.srt
#Traditional replace operation does not do anything to the string it is operating on.
#It only returns are replaced version of the string. We want to meddle with the string
#we are operating on.
    def string_word_count(self):
        print self.str.find(' ', 0, len(self.str) - 1) + 1
    def string_add(self, sr):
        self.str = self.str+sr
        print self.str

str = string('Hello my name is Slim Shady')
str.string_count('a')
str.string_endswith('d')#Answers: does there exist a substring ending in d
str.string_find('na', 0, len(str.str) - 1)#Returns the starting index
str.string_replace('Slim', 'Fat', 12)
str.string_word_count()
str.string_add('. Eminem\'s eating M&Ms.')
str.string_replace('. Eminem\'s eating M&Ms.','',1)#This is how we can delete substrings.
#Replace a substring with ''.
