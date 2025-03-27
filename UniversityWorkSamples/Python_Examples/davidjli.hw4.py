# Stats701 Homework 4
# Name: David Li

## Part 1: Still More Fun With Vectors
## (Spent about 4 hours)

# Q1: Defining a class Vector

class Vector:
    '''Represents a Vector'''
    def __init__(self, dim = 0, entries = ()): # Initiation of class attributes, these are defaults
        self.dim = dim
        self.entries = entries

        if (type(self.dim) != int): # Wrong Type
            raise TypeError("Dimensions should be a non-negative integer!")
        if (self.dim < 0): # Negative Dimensions
            raise ValueError("Dimensions should be a non-negative integer!")
        if (self.dim >= 0 and self.entries == ()): # Dimensions supplied, entries not supplied
            for x in range(0, self.dim): # Make a blank tuple with same number of 0's as dimension number supplied
                self.entries = self.entries + (0,)
        if (self.dim != len(self.entries)):
            raise IndexError("Dimensions supplied don't match length of entries supplied!")

    def print_vec(self): # For debugging
        print(self.entries)

    def dot(self, vector): # Q4
        if not isinstance(vector, Vector):
            raise TypeError("The input vector needs to be of class Vector")
        if (len(vector.entries) != len(self.entries)):
            raise IndexError("Dimensions of the starting vector and input vector need to match")
        innerprod = sum(float(index1) * float(index2) for (index1, index2) in zip(self.entries, vector.entries))  # Product
        return innerprod

    def __mul__(self, other): # Q5
        if isinstance(self, Vector):
            if (type(other) == int or type(other) == float):
                new = Vector()
                new.entries = tuple([other*x for x in self.entries])
                new.dim = len(new.entries)
                return new
            elif (isinstance(other, Vector)):
                if (len(self.entries) != len(other.entries)):
                    raise IndexError("Vector-to-vector multiplication requires equal lengths of the two vectors")
                else:
                    new = Vector()
                    new.entries = tuple(float(index1) * float(index2) for (index1, index2) in zip(self.entries, other.entries))
                    new.dim = len(new.entries)
                    return new
            else:
                raise TypeError("Multiplication overloading does not support multiplying these types!")

    __rmul__ = __mul__

    def norm(self, p): # Q6
        if p == 0:
            counter = 0
            for i in self.entries[0:len(self.entries)]:
                if i != 0:
                    counter = counter + 1
            return counter
        elif p == float('Infinity'):
            counter = 0
            for i in self.entries[0:len(self.entries)]:
                if abs(i) > counter:
                    counter = abs(i)
            return counter
        elif p > 0:
            thesum = 0
            for i in self.entries[0:len(self.entries)]:
                thesum = thesum + ((abs(i))**p)
            modifiedsum = thesum**(1/p)
            return modifiedsum
        else:
            raise ValueError("Input for p was probably negative, but needs to be non-negative by implementation definition")

# Test cases that should work
a = Vector(dim = 2, entries = (1,2)) # Normal case
a.print_vec()
b = Vector(2) # Only supply dimensions, expect (0,0)
b.print_vec()
c = Vector(dim = 2, entries = [1.2,2.4]) # Should work with lists and floats too, expect [1.2, 2.4]
c.print_vec()
g = Vector() # Default Behavior
g.print_vec()
e = Vector(dim = 2, entries = (1.2, 2.4)) # Another example
e.print_vec()

# Following test cases will raise errors if uncommented, should raise correct ones
#d = Vector(-1) # Negative Dimensions
#d.print_vec()
#e = Vector("string") # Dimensions supplied is not a integer type
#e.print_vec()
#f = Vector(2, (1,)) # Dimensions supplied don't match the number of entries supplied
#f.print_vec()

# Q2: Entries tuple or list?

# I chose tuples because tuples are immutable by natural property, this limits unintended changes and behavior if there are rare cases (but by natural intuition probably less flexible than lists due to immutability)

# Q3: Class vs Instance Attributes?

# We are using instance attributes, which by implementation is advantageous over class attributes in our case because the entries we desire to be stored will change on instantiation to instantiation basis.
# Though class attributes ensure more consistency, the dimensions and entries need to be initialized differently each time so it is needed to be instance attributes in this case.

# Q4: Vector.dot

# Refer to the class definition in Q1 for implementation, test cases below

print(a.dot(b)) # Should return 0, a is a Vector
print(a.dot(e)) # Should return 6.0, a is a Vector
# print(a.dot(g)) # Unequal dimensions, should be an error

# Q5: Scalar Multiplying

# Refer to the class definition in Q1 for implementation, Many test cases to follow

test = a*e
print(isinstance(test, Vector))
test.print_vec() # Should return (1.2, 4.8)

test1 = a*2
print(isinstance(test1, Vector))
test1.print_vec() # Should return (2, 4)

test2 = 2*a
print(isinstance(test2, Vector))
test2.print_vec() # Should return (2, 4)

#test3 = a*g # Should be an error from mismatch in dimensions
#test3.print_vec()

# Q6: Vector.norm

# Refer to the class definition in Q1 for implementation, test cases below

print(a.norm(0)) # Expect 2
print(a.norm(1)) # Expect 3
print(a.norm(1.2)) # Should work with floats
print(a.norm(float('Infinity'))) # Expect 2
#print(a.norm(-1)) # Expect error

print(b.norm(0)) # Expect 0
print(b.norm(1.2)) # Expect 0
print(b.norm(float('Infinity'))) # Expect 0

## Part 2: Objects and Classes Geometry Edition
## (Spent about 2 hours)

# Q1: Defining a class Point

class Point:
    '''Defines a point in 2-D Euchlidean Space'''
    def __init__(self, xval = 0, yval = 0): # Initializes a point, default point is the origin
        self.xval = xval
        self.yval = yval

    def printapoint(self):
        print((self.xval, self.yval))

    #Q2:

    def __lt__(self, point):
        if isinstance(point, Point):
            if (self.xval < point.xval):
                return True
            elif (self.xval == point.xval):
                if (self.yval < point.yval):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")
    def __le__(self, point):
        if isinstance(point, Point):
            if (self.xval > point.xval):
                return False
            elif (self.xval == point.xval):
                if (self.yval <= point.yval):
                    return True
                else:
                    return False
            else:
                return True
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")
    def __gt__(self, point):
        if isinstance(point, Point):
            if (self.xval > point.xval):
                return True
            elif (self.xval == point.xval):
                if (self.yval > point.yval):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")
    def __ge__(self, point):
        if isinstance(point, Point):
            if (self.xval < point.xval):
                return False
            elif (self.xval == point.xval):
                if (self.yval >= point.yval):
                    return True
                else:
                    return False
            else:
                return True
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")
    def __eq__(self, point):
        if isinstance(point, Point):
            if ((self.xval == point.xval) and (self.yval == point.yval)):
                return True
            else:
                return False
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")
    def __ne__(self, point):
        if isinstance(point, Point):
            if ((self.xval == point.xval) and (self.yval == point.yval)):
                return False
            else:
                return True
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")

    #Q3:

    def __add__(self, point):
        if isinstance(point, Point):
            newxval = self.xval + point.xval
            newyval = self.yval + point.yval
            newPoint = Point(xval = newxval, yval = newyval)
            return newPoint
        else:
            raise TypeError("Type of Input needs to be a Point for comparison")



# Q2: Point Comparison

# See Class Definition in Q1, test cases below

p1 = Point(2,3)
p2 = Point(-2,4)
p3 = Point(2,5)
p4 = Point(3,5)

# True cases
print(p1 < p3)
print(p2 <= p2)
print(p4 > p3)
print(p4 >= p4)
print(p2 == p2)
print(p2 != p3)
# False cases
print(p4 <= p3)
print(p4 < p1)
print(p1 > p4)
print(p2 >= p3)
print(p2 == p3)
print(p2 != p2)

# Q3: Addition of points

# See Class Definition in Q1, test cases below

sum1 = p1 + p3 # Should be (4, 8)
sum1.printapoint()
sum2 = p1 + p2 # Should be (0, 7)
sum2.printapoint()

# Q4: Line implementation

class Line:
    '''Represents a line in 2-D euchlidean plane'''
    def __init__(self, slope = 0, yinter = 0): # Default is a horizontal line at y = 0
        self.slope = slope
        self.yinter = yinter

    def project(self, point):
        if isinstance(point, Point):
            x1 = point.xval - 10 # Simulate a left point on the line
            y1 = self.slope*x1 + self.yinter
            x2 = point.xval + 10 # Simulate a right point on the line
            y2 = self.slope*x2 + self.yinter
            leftp = Point(xval = (x1), yval = (y1)) # Following the equation found online of how to obtain the projection point
            rightp = Point(xval = (x2), yval = (y2))
            px = rightp.xval - leftp.xval
            py = rightp.yval - leftp.yval
            dAB = (px**2 + py**2)
            u = (((point.xval - leftp.xval) * px)  + ((point.yval - leftp.yval) * py)) / dAB
            projx = (leftp.xval + u * px)
            projy = (leftp.yval + u * py)
            projected = Point(xval = projx, yval = projy) # Creating a new point object with projected coordinates
            return projected
        else:
            TypeError("Input needs to be of Point class")

# Q5: Line.project

# See class definition in Q4, test cases below

testline1 = Line(slope = 1, yinter = 0) # y = x  line
testline2 = Line(slope = 0, yinter = 3) # y = 3 line

testpoint1 = Point(xval = 4, yval = 2) # (4, 2) point
testpoint2 = Point(xval = 4, yval = 3) # (4, 3) point
testing = testline1.project(testpoint1)
testing.printapoint() # Expecting (3, 3)
testing2 = testline2.project(testpoint1)
testing2.printapoint() # Expecting (4, 3)

## Part 3: Objects and Inheritance
## (Spent about  minutes)

# Q1: Defining a class Author

class Author:
    '''A class called Author used for the related bibliography implementations that follow'''
    next_id = 0 # An id keeping track of the next queued id

    def __init__(self, given_name = None, family_name = None):
        self.given_name = given_name
        self.family_name = family_name
        self.author_id = Author.next_id
        Author.next_id = Author.next_id + 1 # Increment id

    # Q2
    def __str__(self):
        abbre = self.given_name[0].upper() # This creates the initial of the first name
        sequence = self.family_name + "," + " " + abbre + "."
        return sequence

    def debugg(self): # A debugging function
        print("")
        print(self.author_id)
        print(self.next_id)
        print(self.given_name)
        print(self.family_name)

# Testing
print(Author.next_id) # Initialized and should be 0
run1 = Author(given_name = "Bob", family_name = "Junior") # Add a person called Bob Junior
run1.debugg()

run2 = Author(given_name = "mary", family_name = "Sue") # Add a person called Mary Sue
run2.debugg()

# Q2: __str__ operator Author

# Implemented in the class definition in Q1, test cases are below

print(run1)
print(run2)

# Q3: Defining a class Document

class Document:
    '''Document class containing author, title, and year'''
    def __init__(self, author = [], title = None, year = None):
        self.list = author
        self.title = title
        self.year = year

        if ((type(author) is not list) or (not all(isinstance(counter, Author) for counter in author)) or (author == [])): # Check conditions of the author input
            raise TypeError("Author input should be a non-blank list, filled with Authors of class type")
        else:
            self.author = self.list
        if isinstance(title, str): # Check that title is string
            Document.title = self.title
        else:
            raise TypeError("Title should be in string form!")
        if isinstance(year, int): # Check that year is an integer
            Document.year = self.year
        else:
            raise TypeError("Year should be in integer form!")

    def __str__(self): # For printing
        if ((self.author == None) or (self.title == None) or (self.year == None)): # Ensures that the parameters should be specified
            raise ValueError("The Document author, title, or year needs to be specified!")
        else:
            sequence = ""
            for x in self.author: # Iterates through all the authors in the author list
                sequence = sequence + x.family_name + ", " + x.given_name[0].upper() + ". and "
            sequence = sequence[:-4]
            sequence = sequence + "(" + str(self.year) + ").  " + self.title + "."
            return sequence

# Q4: __str__ operator Document

# Implemented in class definition in Q3, below are test cases testing different lengths in the author list
example1 = Document([Author("David", "Li")], "mytitle", 1920)
print(example1)
example2 = Document([Author("David", "Li"), Author("Bob", "Junior")], "mytitle", 1920)
print(example2)
example3 = Document([Author("David", "Li"), Author("Bob", "Junior"), Author("Joe", "Schmoe")], "mytitle", 1920)
print(example3)

# Q5: class Book inherited off of Document

class Book(Document):
    '''An inherited class off of the Document class'''

    def __init__(self, author = [], title = None, year = None, publisher = None): # Adding publisher
        Document.__init__(self, author, title, year)
        if not isinstance(publisher, str): # Check publisher should be a string
            raise TypeError("Publisher input expected to be type-string.")
        else:
            self.publisher = publisher

    def __str__(self): # For printing
        sequence = Document.__str__(self) + " " + self.publisher + "."
        return sequence

# Testing

book1 = Book([Author("David", "Li")], "mytitle", 1920, "exampublish")
book2 = Book([Author("David", "Li"), Author("Bob", "Junior")], "mytitle", 1920, "exampublish")

print(book1)
print(book2)