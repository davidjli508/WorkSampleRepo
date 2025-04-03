# Stats701 Homework 2
# Name: David Li

# Prelims

## Part 1: Fun With Strings
## (Spent about 30 minutes)

# Q1: Defining a function, taking a string and figuring out if it's a palindrome

def is_palindrome(instring):
    # These lines omit capitalization and spaces, for the comparison later
    instringcaps = instring.lower()
    trimmedstring = instringcaps.replace(" ", "")

    # These lines omit capitalization and spaces for the backwards version, for comparison later
    backwards = instring[::-1]
    backwardslower = backwards.lower()
    trimmedback = backwardslower.replace(" ", "")

    # Time to compare forwards and backwards versions of the input string
    if(trimmedstring == trimmedback):
        return True
    else:
        return False

# Testing
print(is_palindrome("tacocat")) # Tests for normal case
print(is_palindrome("Taco cat")) # Specification's test #2
print(is_palindrome("01234543210")) # Tests for numbers only
print(is_palindrome("1Was it a car or a cat I saw1")) # Tests for spaces and capitalization, and numbers

# Q2: "Abecedarian" function

def is_abecedarian(instring):
    # Omitting the need to worry about spaces
    trimmedstring = instring.replace(" ", "")

    # Keeping track of how many times we found a letter not happening in alphabetical order
    counter = 0
    for letter in range(len(trimmedstring) - 1):
        if (trimmedstring[letter] > trimmedstring[letter + 1]):
            counter = counter + 1

    # If there are any violations, return False. Otherwise True
    if counter > 0:
        return False
    else:
        return True

# Testing
print(is_abecedarian("adder")) # Expecting True
print(is_abecedarian("beet")) # Expecting True
print(is_abecedarian("dog")) # Expecting False
print(is_abecedarian("cat")) # Expecting False
print(is_abecedarian("abcd efgh xyz")) # Expecting True

# Q3: Count Vowels function

def count_vowels(instring):
    # Converting the input string into all lowercase for later comparison ease
    instringcaps = instring.lower()

    # Initialize a counter to store how many vowels and increment if found
    counter = 0
    for letter in range(len(instring)): # Parse the string
        if (instringcaps[letter] in ('a', 'e', 'i', 'o', 'u')):
            counter = counter + 1

    # Whats the total?
    return counter

# Test Cases
print(count_vowels("aeiou")) # Expecting 5
print(count_vowels("a e i o u")) # Expecting 5
print(count_vowels("doggeroni")) # Expecting 4
print(count_vowels("Stats LUL")) # Expecting 2

## Part 2: Fun with Lists
## (Spent about 60 minutes)

# Q1: List_Reverse

def list_reverse(inlist):
    if (isinstance(inlist, list) == False): # Checking if the input is a list, otherwise error
        raise TypeError("The input needs to be a list")
    else:
        outputlist = inlist[::-1] # Reverse the list
        return outputlist

# Test Cases
print(list_reverse([1, 2, 3]))
print(list_reverse([1, "a", "b"])) # Mixed Types
print(list_reverse([["chunk1"], [1, 2, 3]])) # Lists within lists

# Q2: Binary search

def binary_search(t, elmt): # Assume the list is sorted integers
    # Variables of note
    length = len(t)
    midpoint = ((length-1) // 2) # Floor Division

    # Base case of empty list, and also terminates the recursion
    if length == 0:
        return False
    else:
        if (t[midpoint] == elmt): # Checking the midpoint
            return True
        elif (t[midpoint] > elmt): # If the midpoint is greater than the desired number
            return binary_search(t[:(midpoint)], elmt) # Look at the lower half and use recursion
        else:
            return binary_search(t[(midpoint + 1):], elmt) # otherwise look at upper half and use recursion

# Testing
testeven = [1, 2, 3, 4, 5, 6] # Length is 6
testodd = [1, 2, 3, 4, 5] # Length is 5
# These should print True
print(binary_search(testeven, 1))
print(binary_search(testeven, 2))
print(binary_search(testeven, 3))
print(binary_search(testeven, 4))
print(binary_search(testeven, 5))
print(binary_search(testeven, 6))
print(binary_search(testodd, 1))
print(binary_search(testodd, 2))
print(binary_search(testodd, 3))
print(binary_search(testodd, 4))
print(binary_search(testodd, 5))
# These should print False
print(binary_search(testeven, 7))
print(binary_search(testodd, 6))


## Part 3: More Fun With Strings
## (Spent about 45 minutes)

# Q1: char_hist function

def char_hist(instring):
    # Converting all characters to lowercase, since lowercase and capitals are keyed into lowercase form
    instringcaps = instring.lower()

    # Make the dictionary, parse each letter and store frequency of occurrence into dictionary
    counters = dict()
    for letter in instringcaps:
        counters[letter] = counters.get(letter, 0) + 1
    return counters

# Testing
print(char_hist("gattaca")) # Normal case
print(char_hist("GaTtAca")) # Capitals and lowercase are treated similarly, and keyed on lowercase
print(char_hist("ERROR!!?!?!")) # Special characters handling
print(char_hist("crazy number 2 3 4")) # Spaces and numbers handling
print(char_hist("    ")) # Multiple Space handling

# Q2: bigram_hist function

def bigram_hist(instring):
    # Converting all characters to lowercase, since lowercase and capitals are keyed into lowercase form
    instringcaps = instring.lower()

    # Make the dictionary, parse each bigram and store frequency of occurrence into dictionary
    counters = dict()
    for x in range(len(instringcaps) - 1): # "-1" is due to we don't want the last single letter as the bigram will only have one letter
        bigram = instringcaps[x : x + 2] # Extract 2 letters
        counters[bigram] = counters.get(bigram, 0) + 1
    return counters

print(bigram_hist("mississippi")) # Normal case
print(bigram_hist("miSsissIppI")) # Capitals and lowercase treated similarly
print(bigram_hist("cat, dog")) # Spaces handling
print(bigram_hist("wow!?! !!??")) # Special characters handling
print(bigram_hist("     catdog")) # Multiple Space Handling

## Part 4: Tuples as Vectors
## (Spent about 2 hours)

# Q1: vec_scalar_multi

# Helper Boolean function to verify an input is a numerical tuple of all floats and/or ints
def num_tuple(intuple):
    if type(intuple) is tuple:
        counter = 0
        for i in range(len(intuple)): # iterating through the tuple
            if type(intuple[i]) in (float, int): # counts number of floats or ints
                counter = counter + 1
        if (counter == len(intuple)): # ensures all elements were float or int
            return True
        return False
    else:
        return False

def vec_scalar_mult(t, s):
    if ((num_tuple(t) == True) and (type(s) in (float, int))): # Verify the types of the inputs are correct
        newtuple = tuple([s*x for x in t]) # Creating a new tuple with the multiplication done
        return newtuple
    else:
        raise TypeError("Tuple and Scalar must be numerical")

# Test Cases
# These should work
print(vec_scalar_mult((2, 4.4), 2))
print(vec_scalar_mult((2.2, 3), 3.3))
# Below are error cases
# print(vec_scalar_mult(2, 3))
# print(vec_scalar_mult("stre", 2))
# print(vec_scalar_mult((2, 4), "test"))

# Q2: Vector inner products

def vec_inner_product(tuple1, tuple2):
    if ((num_tuple(tuple1) == True) and (num_tuple(tuple2) == True)): # Verify the tuples are valid through earlier helper function
        if(len(tuple1) == len(tuple2)): # Enforce equal lengths
            innerprod = sum(float(index1) * float(index2) for (index1, index2) in zip(tuple1, tuple2)) # Product
            return innerprod
        else:
            raise ValueError("The tuple lengths must be equal to do inner product")
    else:
        raise TypeError("Both Tuples need to be numerical")

# Testing
print(vec_inner_product((1,2), (3, 4))) # Expecting 11 as output

# Q3: check_valid_mx

def check_valid_mx(input):
    if type(input) is tuple: # Input is a tuple
        fixedlength = len(input[0]) # Length that all the sub-tuples should be
        for i in range(len(input)): # Checking each tuple..
            if ((len(input[i]) != fixedlength) or (num_tuple(input[i]) == False)): # If subtuples are truly numerical tuples and are of same length
                return False
        return True
    else:
        return False

# Test Cases
testtuple1 = ((1,2), (3,4), (5, 6))
testtuple2 = ((1,2), "hi", (5, 6))
testtuple3 = ((1,2), (1,2,3))

# First should be true, rest should be false
print(check_valid_mx(testtuple1))
print(check_valid_mx(testtuple2))
print(check_valid_mx(testtuple3))

# Q4: mx_vec_mult

def mx_vec_mult(mat, vec):
    if ((check_valid_mx(mat) == True) and (num_tuple(vec) == True)): # Ensure matrix and tuples are valid
        newlist = []
        for i in range(len(mat)): # iterate through each subtuple in the tuple of subtuples
            if (len(mat[i]) != len(vec)): # Dimensions need to be correct for matrix multiplication
                raise ValueError("Incorrect Dimensions")
            newlist.append(vec_inner_product(mat[i], vec)) # Use inner product as the matrix multiplication, and append the result
        finaltuple = tuple(newlist) # Should be a tuple of post matrix multiplication
        return finaltuple
    else:
        raise TypeError("Needs a valid matrix and valid tuple as inputs")

# Test Case (Assuming valid matrix and tuple, which should be valid from proper implementation and test-case checking in the last few parts
testmat = ((1,2,3), (4,5,6), (7,8,9))
testvec = tuple((2,2,2))

print(mx_vec_mult(testmat, testvec)) # Expected output should be (12, 30, 48)