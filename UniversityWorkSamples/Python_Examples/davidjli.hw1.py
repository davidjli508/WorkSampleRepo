# Stats701 Homework 1
# Name: David Li

# Prelims
import math

## Part 1: Defining Simple Functions
## (Spent about 10 minutes)

# Q1: Defining a function, with no input and prints the "Hello World" string
def say_hello():
    print('Hello, world!')

say_hello()

# Q2: Defining a function goat_pad which takes a string as input and adds "goat" to beginning and end
def goat_pad(string):
    answer = ('goat' + string + 'goat')
    print(answer)
    return(answer)

goat_pad('test')
goat_pad('bird')
goat_pad('_')

# Q3: Defining a function print_n which takes two arguments, a string and a integer and prints the string n times on separate lines
def print_n(s, n):
    counter = 0
    while counter != n:
        print(s)
        counter = counter + 1

print_n('test', 5)
print_n(s = '0', n = 3)

## Part 2: Euchlid's algorithm
## (Spent about 15 minutes)

# Q1: Implementing Euchlid's algorithm from wikipedia psuedocode
def gcd(int1, int2):
    if int2 == 0:
        return int1
    else:
        return gcd(int2, int1 % int2)

# Q2: Testing the implementation
print(gcd(20, 10)) # Expect 10 as output
print(gcd(2017, 2018)) # Expect 1 as output
print(gcd(1000, 250)) # Expect 250 as output
print(gcd(5040, 60)) # Expect 60 as output

# Q3: Conceptual Question
print(gcd(-2, 6)) # Expect 2 as output
print(gcd(6, -2)) # Expect 2 as output
print(gcd(-2, -6))
print(gcd(-6, -2))

# If both numbers are negative but there is a valid gcd, the function will not work as it does not account for positive values
# If one number is negative, the ordering of which integer in the input is negative will matter
# due to how the algorithm is implemented (it has one number modulus divided by the other). Results could be correct or wrong
# depending on ordering

## Part 3: Approximating Euler's number e
## (Spent about 1 hour)

# Will need a factorial function
def own_factorial(n):
    sum = 1
    while(n != 0):
        sum = sum * n
        n = n - 1
    return(sum)

print(own_factorial(0))
print(own_factorial(1))
print(own_factorial(4))

# Q1: Implementing euler_limit, based on a limit approach to euler's number
def euler_limit(n):
    result = (1 + (1 / n)) ** n
    return(float(result)) # Output should be a float number

print(euler_limit(1)) # Expected output: 2
print(euler_limit(5)) # Expected output: 2.48832

# Q2: Implementing euler_infinite_sum, prints first few terms' sum
def euler_infinite_sum(n):
    sum = 0
    counter = 0
    while(counter != n):
        equation = 1 / own_factorial(counter)
        sum = sum + equation
        counter = counter + 1
    return(float(sum))

print(euler_infinite_sum(0)) # Expect 0 as output
print(euler_infinite_sum(1)) # Expect 1 as output
print(euler_infinite_sum(4)) # Expect 2.667 as output

# Q3: Implementing euler_approx
def euler_approx(epsilon):
    if(type(epsilon) != float):
        print('Input is not a float!')
    else:
        lowerbound = (math.exp(1) - epsilon)
        upperbound = (math.exp(1) + epsilon)
        sum = 0
        counter = 0
        while (sum <= lowerbound or sum >= upperbound):
            equation = 1 / own_factorial(counter)
            sum = sum + equation
            counter = counter + 1
            if(sum >= math.exp(1)):
                print('Something bad happened..')
                break
        return(float(sum))

# These test cases should all return true...
print(euler_approx(0.5) > (math.exp(1) - 0.5))
print(euler_approx(.05) > (math.exp(1) - 0.05))
print(euler_approx(.005) > (math.exp(1) - 0.005))

# Q4: Implementing print_euler_sum_table
def print_euler_sum_table(n):
    if(type(n) == int and n > 0):
        for k in range(1, n + 1):
            print(euler_infinite_sum(k))
    else:
        print('Input is not a positive integer')

# Test output, the two implementations should be the same
print_euler_sum_table(5)
# vs
print(euler_infinite_sum(1))
print(euler_infinite_sum(2))
print(euler_infinite_sum(3))
print(euler_infinite_sum(4))
print(euler_infinite_sum(5))
