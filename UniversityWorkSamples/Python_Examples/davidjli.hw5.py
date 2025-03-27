# Stats701 Homework 5
# Name: David Li

## Part 1: Iterators and Generators
## (Spent about 1 hour)

# Q1: Defining a class Fibo of iterators

class Fibo:
    '''Represents a class for iterating over fibonacci numbers'''
    def __init__(self, stop): # Initialization of sequence of fibonacci numbers
        self.stop = stop # How many fibonacci numbers you want
        self.counter = 1 # Keeps track of how many fibonacci numbers have been given
    def __iter__(self): # Keeps track of 2 numbers to add through the increments
        self.start = 0
        self.after = 1
        return self
    def __next__(self):
        current = self.start
        if self.counter <= self.stop: # Stop if we have the number of fib numbers we want
            temp = self.start # Incrementing the numbers
            self.start = self.after
            self.after = temp + self.after
            self.counter = self.counter + 1
        else:
            raise StopIteration("Has iterated until the stop value")
        return current

# Testing
for n in Fibo(10): # Expecting 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    print(n, end = " ")
print("")

# Q2: integers generator

def integers(): # Infinite incrementation of integers starting from 0
    counter = 0
    while True:
        yield counter
        counter = counter + 1

# Testing

test = integers()
print("")
print(test) # This should be a generator object
print([next(test) for _ in range(10)]) # Expecting 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

# Q3: primes generator

def primes():
    counter = 2 # First prime number
    while True:
        switch = True
        for divisor in range(2, counter):
            if (counter % divisor == 0): # If we find a number to evenly divide into except 2 and the number itself, this number is out of consideration
                switch = False
                break
        if switch == True: # Otherwise yield the number
            yield counter
        counter = counter + 1

# Testing

test2 = primes()
print("")
print(test2) # Should be a generator object
print([next(test2) for _ in range(10)]) # Expecting 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
print("")


## Part 2: List Comprehensions and Generator Expressions
## (Spent about 1.5 hours)

# Q1: list comprehension of odd squares of integers 1-20
def squaring(x): # Function for calcuating the squares
    return (x**2)
print([int(squaring(x)) for x in range(1, 21) if (x%2 != 0)]) # Applying squaring function to 1...20 and for odd numbers only

# Q2: gen expression for perfect cubes
cubes = (x**3 for x in integers() if x >= 1)
print("")
print(cubes) # Expecting a generator object
print(next(cubes)) # Expecting 1
print(next(cubes)) # Expecting 8
print(next(cubes)) # Expecting 27

# Q3: gen expression for tetrahedral numbers
def bin(n, x): # Self code for calculating the binomial coefficient for n choose x
    ways = 1
    for z in range(1, min(x, n - x) + 1):
        ways = ways * n
        ways = ways // z
        n = n - 1
    return ways

tetra = (bin((x+2), 3) for x in integers() if x >= 1)
print("")
print(tetra) # Expecting a generator object
print(next(tetra)) # Expecting 1
print(next(tetra)) # Expecting 4
print(next(tetra)) # Expecting 10
print(next(tetra)) # Expecting 20
print(next(tetra)) # Expecting 35
print(next(tetra)) # Expecting 56

tetra2 = (bin((x+2), 3) for x in integers() if x >= 1) # Separate generator for "resetting" generator state for tetra numbers in Part 3

## Part 3: Map, Filter, Reduce
## (Spent about 3 hours)

from itertools import accumulate
from functools import reduce

# Q1: Sum of first 10 odd square numbers
# Question is awkward, providing two possible interpretations

# Interpretation 1, provide the sum of the squares of the odd numbers between 1 and 10
sumoddsquares1 = sum(map(lambda x: (x**2) if (x % 2 != 0) else 0, range(1, 11)))
print("")

# Below these two should match
print(sumoddsquares1)
print(1**2 + 3**2 + 5**2 + 7**2 + 9**2) #+ 11**2 + 13**2 + 15**2 + 17**2 + 19**2)

# Interpretation 2, provide the sum of the squares of the first 10 odd numbers
sumoddsquares2 = sum(map(lambda x: (x**2) if (x % 2 != 0) else 0, range(1, 21))) # Between 1 and 10 has 5 odd numbers, 11 and 20 has 5 odd numbers for total of 10
print("")

# Below these two should match
print(sumoddsquares2)
print(1**2 + 3**2 + 5**2 + 7**2 + 9**2 + 11**2 + 13**2 + 15**2 + 17**2 + 19**2)

# Q2: Product of first 17 primes
first17prime = reduce(lambda x,y: x*y, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]) # Continuously multiply through the first 17 primes, return the total product
print("")

# Below these two should match
print(first17prime)
print(2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53*59)

# Q3: First ten harmonic numbers
harmonics10 = list(accumulate((1/h) for h in range(1, 11))) # Accumulate function is useful, keeps a running track of the sums
print("")

# Below these two should match
print(harmonics10)
print([1.0, 3/2, 11/6, 25/12, 137/60, 49/20, 363/140, 761/280, 7129/2520, 7381/2520]) # The true harmonic numbers

# Q4: geometric mean of first 10 tetrahedral numbers
tetrageomean = ((reduce(lambda x, y: x*y, list(next(tetra2) for _ in range(0,10)))) ** (1/10)) # Reduce multiplies through the values, then finally raise to (1/10)
print("")

# Below these two should match
print(tetrageomean)
print("29.91378181")

## Part 4: Fun with Polynomials
## (Spent about 2.5 hours)

# Q1: eval_poly

def eval_poly(x, coeffs): # This is representation of the polynomial formula, reduce takes a running sum through all of the coefficients
    return(reduce(lambda x, y: x+y, (coeffs[z] * (x**z) for z in range(len(coeffs)))))
print("")

# Below these two should match
print(eval_poly(3, [1,2,3]))
print((1*(3**0)) + (2*(3**1)) + (3*(3**2)))

# Q2: make_poly

def make_poly(coeffs): # Lambda is considered to operate as a function, so we have make_poly with coefficients that contains it's own lambda function to take a x parameter and use our eval_poly function
    return (lambda x: eval_poly(x, coeffs))
print("")

# Below these two should match
print(make_poly([2,3,4])(3))
print((2*(3**0)) + (3*(3**1)) + (4*(3**2)))