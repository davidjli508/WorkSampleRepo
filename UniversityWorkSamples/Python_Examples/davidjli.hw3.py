# Stats701 Homework 3
# Name: David Li

# Collaborated with Anuj Dahiya

# Prelims
import pickle

## Part 1: More Fun With Tuples
## (Spent about 60 minutes)

# Q1: my_sum function

def my_sum(*args):
    if len(args) == 0: # Taking care of base case of no arguments, sum = 0
        totalsum = 0
    else:
        totalsum = sum(args)
    return totalsum

# Test Cases
print(my_sum()) # Expecting 0 as output
print(my_sum(1,2,3,4,5)) # All integers, expecting 15 as output
print(my_sum(1.2,2.3,3.4,4.5)) # All floats, expecting 11.4 as output
print(my_sum(1,2.2,3.4,5)) # Mix of ints and floats, expecting 11.6 as output

# Q2: reverse_tuple function

def reverse_tuple(tuple):
    newtuple = tuple[::-1] # Reverses the tuple
    return newtuple

# Test Cases
print(reverse_tuple((1,2,3,4,5))) # Expecting 5,4,3,2,1 as output
print(reverse_tuple(('a', 'b', 'c'))) # Expecting 'c' 'b' 'a' as output
print(reverse_tuple(('a', 1, 2))) # Expecting 2, 1, 'a' as output
print(reverse_tuple(('a',))) # Test case for length one tuple, expecting 'a' as output

# Q3: rotate_tuple function

def rotate_tuple(intuple, n):
    if type(intuple) is tuple: # Ensures input "tuple" is really a tuple
        try:
            m = (n % len(intuple)) # Find out how many spaces you actually are shifting, it's a multiple
            return intuple[-m:] + intuple[0:-m] # Make a new tuple with the entries rearranged
        except TypeError: # If we have a TypeError, we coerce to integer and print a message
            n = int(n)
            print("Input was not as expected, should be an integer")
            return rotate_tuple(intuple, n) # Call the function again, but this time use an coerced integer
        except ZeroDivisionError:
            print("Your tuple was length 0, printing the empty tuple:") # Special case, where if the input tuple is empty we'll just print an empty tuple
            return ()
    else:
        raise TypeError("The first input needs to be a tuple!")

# Testing

# These should give Input Errors for the tuples inputted
# print(rotate_tuple([1,2,3], 2)) # This is a list, should raise TypeError of invalid tuple

# A empty tuple is still a tuple, so we just print a message that an empty tuple was inputted and return the empty tuple
# print(rotate_tuple((), 2))

print(rotate_tuple((1,2,3), 1)) # Should have output as (3,1,2)
print(rotate_tuple((1,2,3), 1.5)) # Should have output as if n = 1, and match previous line

print(rotate_tuple((1,2,3), 4)) # Should wrap properly, output as (3,1,2)
print(rotate_tuple((1,2,3), 4.5)) # Should have output as if n = 3, and match previous line

print(rotate_tuple((1,2,3), -1)) # Should have output as (2,3,1)
print(rotate_tuple((1,2,3), -1.5)) # Should have output as if n = -1, and match previous line

print(rotate_tuple((1,2,3), -4)) # Should have output as (2,3,1)
print(rotate_tuple((1,2,3), -4.5)) # Should have output as if n = -4, and match previous line


## Part 2: More Fun With Vectors
## (Spent about 90 minutes)

# Q1: Working with sparse vectors - is_valid_sparse_vector
def is_valid_sparse_vector(valid_or_no):
    if (type(valid_or_no) is dict):
        switch = True # Start with a flagged default of true
        zeroes = 0
        for key, value in valid_or_no.items(): # iterate through the dictionary
            if (type(key) is not int) or (key <= 0): # Ensure all indices are positive integers, otherwise trip the flag
                switch = False
            if (type(value) is not float): # All values must be floats, otherwise trip the flag
                switch = False
            if (value == 0.0):
                zeroes = zeroes + 1
        if ((zeroes == len(valid_or_no.items())) or (len(valid_or_no) == 0)):# If the vector only has values of 0 or if the starting vector was empty, this will happen.
            switch = False # There are no indices or values, so they technically do not satisfy the respective requirements of being positive integers and being floats
        return switch
    else:
        raise TypeError("The input should be a dictionary type!")

# Testing
def dictionaryform1(vector): # Helper function to assist testing
    counters = dict((key,value) for key, value in (enumerate(vector, 1)) if value != 0) # 1-indexed for keys
    return counters
def dictionaryform2(vector): # Helper function to assist testing
    counters = dict((key,value) for key, value in (enumerate(vector))) # 0-indexed for keys
    return counters

# True Cases
# Normal Case
print(is_valid_sparse_vector(dictionaryform1((1.0, 2.0, 3.0)))) # Normal Case
print(is_valid_sparse_vector(dictionaryform1((1.0, -1.0, 0.0)))) # Normal Case #2

# False Cases
# print(is_valid_sparse_vector(dictionaryform1(5))) # Input needs to be a dictionary
print(is_valid_sparse_vector(dictionaryform1(tuple()))) # Empty tuple, which technically has no indices or values
print(is_valid_sparse_vector(dictionaryform1((0.0, 0.0, 0.0, 0.0)))) # Tuple of 0's, which technically then has no indices due to no non-zero values
print(is_valid_sparse_vector(dictionaryform2((1.0, 2.0, 3.0)))) # Indices are 0-indexed
print(is_valid_sparse_vector(dictionaryform1((1.0, 2, 'a')))) # Values are mixed type

# Q2: sparse_inner_product

def sparse_inner_product(sparse1, sparse2):
    innerprod = 0 # initialize sum at 0
    if ((is_valid_sparse_vector(sparse1) == False) or (is_valid_sparse_vector(sparse2) == False)): # Make sure both are valid sparse
        raise TypeError("One of the inputs is not a valid sparse vector!")
    else:
        for (key1, value1) in sparse1.items(): # iterating through dictionary 1
            for (key2, value2) in sparse2.items(): # iterating through dictionary 2
                if key1 == key2: # if the keys match / are worth multiplying and contributing to the total inner product
                    innerprod = innerprod + (value1 * value2)  # Only adding to inner product total if the key was found in both dictionaries
        return innerprod

# Testing
test1 = dictionaryform1((1.1, 2.345, 0.00, 3.14, 5.1)) # Valid sparse
test2 = dictionaryform1((1.0, 0.00, 2.0, 3.0, 0.0)) # Valid sparse
test3 = dictionaryform1((0.0, 0.0, 0.0, 0.0, 0.0)) # Not valid sparse
test4 = dictionaryform1((1, 2, 3, 4, 5, 1, 2, 3)) # Not valid sparse

# True
print(sparse_inner_product(test1, test2)) # Expect 10.52

# False
# print(sparse_inner_product(test1, test3)) # test3 is not a valid sparse
# print(sparse_inner_product(test3, test4)) # Both are not valid sparses


## Part 3: Counting Word Bigrams
## (Spent about 4 hours)

# Q1: count_bigrams_in_file

def count_bigrams_in_file(filename):
    if type(filename) is not str:
        raise TypeError("Input needs to be a string!")
    else:
        badobj = open("/home/david/Desktop/School/Stats701/Homework3/badstring.txt") # Looking at the bad encoding characters
        badchars = badobj.readline() # taking them into a string
        badchars = badchars + ".," # Also need to handle period and commas
        badobj.close()

        mydict = dict() # Dictionary for the bigrams
        totalstring = '' # Empty string to store file contents into a string for parsing
        with open(filename, 'r') as obj: # Read the file
            for line in obj:
                for word in line.split(): # For each word in each line, lowercase it and concatenate it to the "counter" string without excess punctuation
                    word = word.lower()
                    totalstring = totalstring + word.strip(badchars) + " "
        totalstring = totalstring.strip(' ') # Remove the trailing blank due to this procedure
        totalstring = totalstring.split() # Converts the string into a list
        for x, y in zip(totalstring[:-1], totalstring[1:]): # Using tuple representation of pairs, and counting them into the dictionary
            bigram = (x, y)
            mydict[bigram] = mydict.get(bigram, 0) + 1
        return mydict

# count_bigrams_in_file(5)  # Not a string
print(count_bigrams_in_file("/home/david/Desktop/School/Stats701/Homework3/test.txt"))  # Non-existent file

# Q2: Pickling mobydick.txt

s = count_bigrams_in_file("/home/david/Desktop/School/Stats701/Homework3/mobydick.txt") # Count bigrams for the whole mobydick text
t = open("mb.bigrams.pickle", "wb" ) # Create a file to pickle the results into
pickle.dump(s, t)
t.close() # Close file for good measure

# Q3: Collocations

def collocations(filename):
    if type(filename) is not str: # Check the input was a string
        raise TypeError("Input needs to be a string!")
    else: # take care of the bad characters
        badobj = open("/home/david/Desktop/School/Stats701/Homework3/badstring.txt")  # Looking at the bad encoding characters
        badchars = badobj.readline()  # taking them into a string
        badchars = badchars + ".,"  # Also need to handle period and commas
        badobj.close()

        mydict = dict() # Creating a dictionary for the collocations
        totalstring = ''  # Empty string to store file contents into a string for parsing
        with open(filename, 'r') as obj:  # Read the file
            for line in obj:
                for word in line.split():  # For each word in each line, lowercase it and concatenate it to the "counter" string without excess punctuation
                    word = word.lower()
                    totalstring = totalstring + word.strip(badchars) + " "
        totalstring = totalstring.strip(' ')  # Remove the trailing blank due to this procedure
        totalstring = totalstring.split()  # Converts the string into a list

        counterlen = 1 # iterator through the words
        for x in totalstring[0: len(totalstring)]: # Using tuple representation of pairs, and counting them into the dictionary
            if counterlen == 1: # looking at the first word, store the value in the dictionary after
                forwardword = totalstring[counterlen:counterlen + 1]
                if x in mydict:
                    mydict[x].extend(forwardword)
                else:
                    mydict[x] = forwardword
            elif counterlen == len(totalstring): # looking at the last word, if the word before is not a duplicate then add it
                backwardword = totalstring[counterlen - 2: counterlen - 1]
                if x in mydict:
                    mydict[x].extend(backwardword)
                else:
                    mydict[x] = backwardword
                break
            else: # otherwise add the word in front and back as vlues
                backwardword = totalstring[counterlen - 2: counterlen - 1]
                forwardword = totalstring[counterlen: counterlen + 1]
                if x in mydict: # if the key already exists
                    mydict[x].append(''.join(backwardword))
                    mydict[x].append(''.join(forwardword))
                else: # if the key doesn't exist
                    mydict[x] = backwardword
                    mydict[x].append(''.join(forwardword))
            counterlen = counterlen + 1
        new_dict = {a: list(set(b)) for a, b in mydict.items()} # removes duplicates in the unhashable list, and store into another dictionary
        return new_dict

print(collocations("/home/david/Desktop/School/Stats701/Homework3/test.txt"))

# Q4: Collocation function on mobydick.txt

u = collocations("/home/david/Desktop/School/Stats701/Homework3/mobydick.txt") # Count collocations for the whole mobydick text
v = open("mb.colloc.pickle", "wb" ) # Create a file to pickle the results into
pickle.dump(u, v)
v.close() # Close file for good measure