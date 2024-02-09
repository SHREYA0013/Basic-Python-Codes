# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#summation
s=int(input('enter number1'))
t=int(input('enter number2'))
r=s+t
print('sum  is ', r)

#list with loop
mylist=[0,1,2,3,4,5]
num=0
while num in mylist:
    mylist.remove(num+1)
    num=num+1
    print(mylist)
    

#addition
name=str(input('enter your name: '))
print('hi this is me, ', name)


#name printing
input('what is your name: ')
name=input('what is your name')
print(name)
print('hello', name)

name=str(input('what is your name: '))
print(name)
age=int(input('what is your age? '))
print(age)
print('hello','\n my name is', name, '\n i am',age ,' years old')


#type conversion
age_previous = int(input('Enter your age: '))  # Convert the input to an integer
current_age = age_previous + 8  # Add 8 to the age
print('My actual age is:', current_age)  # Print the current age

first=float(input('enter first number: '))
second=float(input('enter second number: '))
sum= first+second
sum= int(sum)
print(sum)
summation=int(first)+int(second)
print('the summation is: ', summation)


#strings
name= 'Prashmita Shreya'
print(name.upper())
print(name.lower())
print(name.find('a'))
print(name.replace('Prashmita','pupu'))
print(name.replace('P','S'))


#Arithmetic operators
print(3*0)
i=2
i=i+i
print(i)
value=2*3+5/2
print(value) #follow BODMAS

print(3>2 or 1>2) #OR makes any one truth as true
print(3>2 or 1>0)
print(0>2 or 1>2)

print(3>2 and 1>2) #AND makes any one FALSE as false and if both then only true
print(3>2 and 5>2)

# AND Result will be true, if both the expressions are true. If any one or both the expressions are false, the result will be false
#OR Result will be true, even if one of the expression is true. If both the expressions are false, the result will be false
#NOT If the expression is true, result will be false and vice versa


#IF-ELSE
AGE= int(input('what is your age? '))
print('AGE= ', AGE)
if AGE>=18:
    print('You are an adult')
    print('\nYou are eligible to give vote')
elif AGE<18 and AGE>3:
    print('\nYou are a school student')
else:
    print('\nYou are a child')
print('\nTHANK YOU')

#CALCULATOR
First_Number=float(input('Enter your first number: '))
print('The first choosen number is: ', First_Number)

Second_Number=float(input('Enter your second number: '))
print('\nThe first choosen number is: ', Second_Number)

if First_Number> 50 and Second_Number> 50:
    print('follow the methods')
    Operator=input('Select any one operation(+,-,*,/,%,**): ')
    if Operator == '+':
        print('The summation is: ', First_Number + Second_Number)
    elif Operator == '-':
        print('The subtraction is: ', First_Number - Second_Number)
    elif Operator== '*':
        print('The multiplication is: ', First_Number * Second_Number)
    elif Operator== '/':
        if Second_Number == 0:
            print('Cannot divide by zero!')
        else:
            print('The divided value is: ', First_Number / Second_Number)
    elif Operator== '%':
        print('The remainder is: ', First_Number % Second_Number)
    elif Operator== '**':
        print('The powered value is: ', First_Number**Second_Number)
        
        
    else:
        print('\nInvalid Operation')
        
    print('\n Operation successful,mathematically')
else:
    print('\nInvalid Operation')
    



#RANGE,LOOPS
numbers= range(4)
print (numbers)

i=1
while i<=100:
     i=i+1
     print(i)

rows = 100  # Number of rows in the triangle
i = 1

while i <= rows:
    # Print spaces to align the numbers in a triangle shape
    print(' ' * (rows - i), end='')
    
    # Print the numbers for the current row
    for j in range(i):
        print(i, end=' ')
        i += 1
    
    # Move to the next line for the next row
    print()

i = 20

while i >= 1:
    print( i * '@')
    i = i - 1

#LIST
marks=[100,90,88]
print(marks)
print(marks[0])
print(marks[-1])
print(marks[0:2])
for score in marks:
    print(score)

#APPEND OPERATION:
grade = ['AA','A+','A','B+','B']
grade.append('C')
print(grade)
grade.insert(6,'F') #0,1,2,3......
print(grade)
print('A' in grade)
print('d' in grade) #if there or not- true/false
print(len(grade))
grade=['A','B','C']
i=0
while i<len(grade):
    print(grade[i])
    i=i+1



numbers=['10','20','30']
numbers.append('40')
numbers.append('50')
print(numbers)
numbers.clear()
print(numbers)


#BREAK-CONTINUE
students=['mingyu','seungkwan','jeonghan','joshua']
for student in students:
    if student== 'seungkwan':
        break; # Using break to exit the loop early
    print(students)
for student in students:
    if student== 'joshua': # Using continue to skip an iteration
        continue;
    print(student)


#TUPLE
MARKS=(98,95,99)
MARKS[0]=100
MARKS = (98, 95, 99)
new_marks = (100,) + MARKS[1:]  # Create a new tuple with the modified value at index 0

print(new_marks)  # Output: (100, 95, 99)

marks=(90,92,95,98,100)
print(marks.count(95))
print(marks.index(98))

#set
grade={'a','b','c'}
for score in grade:
    print(score)
    
# Define a dictionary to store marks for different subjects
marks = {'english': 95, 'chemistry': 78}

# Print the marks for the 'chemistry' subject
print(marks['chemistry'])

# Add marks for the 'physics' subject
marks['physics'] = 97

# Print the updated dictionary with the added 'physics' subject and marks
print(marks)



#FLIGHTS NUMBER AT AIRPORT
flights_initial= 100
flights_landed=int(input('flights landed today= '))#variables are like containers of data
flights_took_off=int(input('flights took off today= '))
current_no_flights= (flights_initial+flights_landed)-flights_took_off
print('no of flights at the airport= ',current_no_flights)


a=int(input('a= '))
b=int(input('b= '))
c=a%b
print('c= ',c)


#defining scalar:
no_of_units=45
mass_of_sun=1.989e30
length_of_china_wall=21196
print(no_of_units)
print(mass_of_sun)
print(length_of_china_wall)



#vectors:
    #importing numpy library as np:
import numpy as np
col_vector=np.array([[5],[9]])
print('the column vector is: \n',col_vector)
print('the shape of the vector is: ',col_vector.shape)

import numpy as np
row_vector=np.array([[5,9,-1,-6]]) #row vector declaration different from colomn
print('the row vector is: ',row_vector)
print('the shape of the vector is: ',row_vector.shape)

import numpy as np
zero_vector_1=np.array([0,0,0])
print('the vector is: ', zero_vector_1)#Illustrating a Zero row vector of dimension 3 using np.array() function
#Illustrating a Zero row vector of dimension 3 using np.zeros() function
zero_vector_2=np.zeros(3)
print(zero_vector_2)

#Illustrating a Zero column vector of dimension 3 using np.zeros() and reshape() function
zero_vector_3=np.array([[0],[0]])
print('\n', zero_vector_3)

#np.zeros(3) creates a 1-dimensional NumPy array of length 3 filled with zeros.& .reshape(3, -1) reshapes the 1-dimensional array into a 3x1 matrix. The -1 is used to automatically infer the size of the second dimension, ensuring that the total number of elements remains the same.
zero_vector_4=np.zeros(3).reshape(3,-1)
print(zero_vector_4)

#one vectors
one_vector=np.array([1,1])
print(one_vector)
one_vector_2=np.ones(3)
print(one_vector_2)

one_vector_3=np.ones(3).reshape(3,-1)
print(one_vector_3)


#INDEXING
import numpy as np
vector_1=np.array([[4],[-1],[9]])
print('\n',vector_1)
print('\n the element at position 1: ',vector_1[1]) #positions are 0,1,2....
print('\n the element at position 0: ',vector_1[-3]) #positions are from right -1,-2,-3....
##Magnitude of vector can be found using function np.linalg.norm() function
print('\n the magnitude is: ', np.linalg.norm(vector_1))


#MATRICES:
import numpy as np
matrix_first=np.array([[23,45],[-67,90]])
print('two dimensional matrix is\n: ',matrix_first,'\n shape of matrix is: ', matrix_first.shape)

matrix_second=np.array([[23,45,9],[-67,6,90],[12,73,0]])
print('three dimensional matrix is\n: ',matrix_second,'\n shape of matrix is: ', matrix_second.shape)

A=np.array([[3,4],[5,6],[-3,-9]])
print('A is a 3by2 matrix:\n ',A, '\n',A.shape )
#selecting an element at row index 2 and column index 1, starts from 0,1,2 rows and 0,1 colomns
print('\n the element at position [2,1]is: ',A[2,1])
print('\n transpose of matrix A:\n ', np.transpose(A))

#diagonal matrix
import numpy as np
diagonal_matrix=np.diag([1,2,3])
print(diagonal_matrix)
diagonal_matrix=np.diag((1,3,2))
print(diagonal_matrix)
matrix_range=np.diag(np.arange(1,6,2))
print(matrix_range)
#Principal diagonal of a diagonal matrix may also have zeros.

#identity matrix: np.eye & np.identity functions
identity_matrix= np.identity(2)
print('the identity matrix is=\n',identity_matrix)
identity_matrix_1=np.eye(4)
print(identity_matrix_1)


#SYMMETRIC: A = AT i.e.  the matrix A is the same as its transpose.
import numpy as np
AA=np.array([[1,2,3],[3,4,-1],[1,-1,1]])
print('AA=\n',AA)
AA_transpose=AA.transpose()
print('AA_transpose:\n',AA_transpose)
comparison=(AA==AA_transpose)#(returns a matrix of boolean compared values) and saving it in a variable comparison
equal_arrays= comparison.all()
print(equal_arrays)



#TRIANGULAR MATRIX:either a lower triangular or an upper triangular matrix.
import numpy as np
#In NumPy, the functions np.tril() and np.triu() are used to create a lower triangular matrix and an upper triangular matrix
lower_triangular_matrix= np.tril([[345,456,89],[23,79,134],[9,34,576]])
print('the lower triangular matrix is=\n',lower_triangular_matrix)

upper_triangular_matrix=np.triu([[344,89789,23],[323,98,976],[24,45,98]])
print('upper triangular matrix is:\n',upper_triangular_matrix)


#exercise of INFOSYS:
import numpy as np
A=np.array([[3,4,5],[2,6,8],[8,9,1]])
print('A:\n',A)
print('upper triangular matrix:\n',np.triu(A))
B=np.triu(A)
C=np.tril(B)
print('the lower triangular matrix:\n',C)
#upper triangular to a lower triangular matrix is a diagonal matrix
def is_diagonal(C):
    return np.allclose(C,np.diag(np.diag(C)))#allclose function check all the values &return whether TRUE
print('Is matrix C diagonal?:', is_diagonal(C))
def is_symmetric(C):
    return np.allclose(C, C.T)
print('is C symmetric?\n', is_symmetric(C))


#ADDITION OF VECTORS:
import numpy as np
vector_1=np.array([[2],[5],[9]])
vector_2=([[3,5,7]])
out=np.add(vector_1,vector_2)
print(out)
#we can use out or output 
vector_1=np.array([[2],[5],[9]])
vector_2=([[3],[5],[7]])
output=np.add(vector_1,vector_2)
print(output)

import numpy as np
vector_1=np.array([[2,5,9]])
vector_2=([[3,5,7]])
out=np.add(vector_1,vector_2)
print(out)

#MATRIX ADDITION:
import numpy as np
M1=np.array([[3,7,9],[2,5,1]])
print('M1:\n',M1)   
M2=np.array([[4,1,2],[6,3,8]])
print('M2:\n',M2)
output=np.add(M1,M2)
print('the result is:\n',output)



matrix1=[[1,2,3],
         [5,6,7],
         [9,8,7]]
matrix2=[[3,4,5],
         [6,7,8],
         [8,9,1]]
result=[[0,0,0],
        [0,0,0],
        [0,0,0]]
for i in range(len(matrix1)):
    for j in range(len(matrix1[0])):
        result[i][j]=matrix1[i][j]+matrix2[i][j]
for row in result:
    print(result)

#matrix multiplication:
import numpy as np
v=np.array([3,4,5]) 
print('\n',v)
s=2
vector_multiplication=v*s#scalar-vector
print('vector\n:',v,'\nscalar=','scalar vector multiplication:\n',vector_multiplication)

import numpy as np
M=np.array([[3,4,5],
            [5,9,1],
            [5,1,5]])
s=2
matrix_multiplication=M*s
print('matrix:\n',M,'scalar:',s,'\nmatrix multiplied:\n',matrix_multiplication)

#inner product:
import numpy as np
vector_1=np.array([[3,3,4]])
print('vector:\n',vector_1)
vector_2=np.array([[3,4,5]])
print('vector 2:\n',vector_2)
inner_product=np.inner(vector_1,vector_2)
print('value of inner product is:\n',inner_product)


#if inner product is zero then it is orthogonal vector:
import numpy as np
S1=np.array([[6],[4],[9],[1]])
print(S1)
S2=np.array([[7],[6],[5],[3]])
print(S2)
result=np.dot(np.transpose(S1),S2)
print("\nresult:",result)

Vector_1 = np.array([[3],[-1],[2]])
Vector_2 = np.array([[2],[4],[-1]])
print('vector_1=\n',Vector_1)
print('vector_2:\n',Vector_2)
trans=np.transpose(Vector_2)
product=np.dot(trans,Vector_1)
print('\nthe answer is:',product)#it is orthogonal

#angles between two vectors:
import numpy as np
def angle_between(vector_1,vector_2):
    dot_pr=vector_1.dot(vector_2)
    norms=np.linalg.norm(vector_1)*np.linalg.norm(vector_2)
    return np.rad2deg(np.arccos(dot_pr/norms))
vector_1=np.array([3,-1,2])
vector_2=np.array([2,4,-1])
print('vector_1:\n',vector_1,'\nvector_2:\n',vector_2)
print('\nthe angle between vectors:\n',angle_between(vector_1, vector_2))
#dot_pr = vector_1.dot(vector_2): Calculates the dot product of the two input vectors vector_1 and vector_2. The dot product of two vectors is the sum of the products of their corresponding elements.
#norms = np.linalg.norm(vector_1) * np.linalg.norm(vector_2): Computes the product of the norms (magnitudes) of the two input vectors using np.linalg.norm, which calculates the Euclidean norm (magnitude) of a vector. The product of the norms represents the denominator in the formula to calculate the cosine of the angle between the vectors.
#Finally, the function doesn't return anything explicitly, so it would typically be followed by a return statement to return the angle in radians. However, in the provided code snippet, the return statement is missing, so the function doesn't return the calculated angle.

#MATRIX_VECTOR MULTIPLICATION:
import numpy as np
matrix=np.array([[23,45,67],[87,65,32],[90,57,43]])
print('matrix:\n',matrix)
vector=np.array([[23],[45],[98]])
print('vector:\n',vector)
result=np.matmul(matrix,vector)
print('\nmultiplication result:\n',result)


#determinants:If the Determinant of a matrix is 0 then the matrix is called a Singular matrix.
import numpy as np
P=np.array([[29,45],
            [67,13]])
print(np.linalg.det(P))

#inverse of matrices;
import numpy as np
B=np.array([[3,4],
            [5,7]])
print('the matrix is:\n',B)
Binv=np.linalg.inv(B)
print('\n the inverse matrix:\n',Binv)
print('\n multiplication of B and B inverse:\n',np.matmul(B,Binv))
print('\n this is a 2 by 2 identity matrix')


#orthogonal matrix:
import numpy as np
A= np.array([[1.0,0.0],
             [0.0,1.0]])    
print('A=\n',A)
comparison_1=np.dot(A.transpose(),A)==np.dot(A,A.transpose()) #Checking for A.AT=AT.A
print('transpose result:\n',comparison_1)
comparison_2=np.dot(A,A.transpose())==np.eye(2)#Checking for A.AT=Identity Matrix
print(comparison_2)
comparison_3=comparison_1==comparison_2
equal_arrays=comparison_3.all()
print('\nA is orthogonal matrix',equal_arrays)


#rank of a matrix:
import numpy as np
D=np.array([[2,3,5],
            [4,6,10]])
print('D:\n',D)
print('\nRank of D: ', np.linalg.matrix_rank(D))
I=np.eye(5)
print('identity matrix:\n',I)
print('\n the rank of identity matrix of dimension 5 by 5:',np.linalg.matrix_rank(I))


B=np.eye(3)
Binv=np.linalg.inv(B)
print('\n the inverse matrix:\n',Binv)


#EXERCISE:INFOSYS:
import numpy as np
v1=np.array([[1],[1],[1]])
print(v1)
v2=np.array([[2],[2],[2]])
print(v2)
left_hand_side= np.dot(v1.transpose(),v2)# Compute the dot product of v1 and v2
print('\n the result of lhs is:\n',left_hand_side)
norm_v1=np.linalg.norm(v1)# Compute the norms of v1 and v2
norm_v2=np.linalg.norm(v2)
print(norm_v1)
print(norm_v2)
right_hand_side=norm_v1*norm_v2# Compute the right-hand side of the Cauchy-Schwarz inequality
print('\n the right hand result:\n',right_hand_side)
# Check if the Cauchy-Schwarz inequality holds
cauchy_schwarz=left_hand_side<=right_hand_side
print('\nis the inequality holds?',cauchy_schwarz)


#EXERCISE:INFOSYS:
import numpy as np
v3=np.array([[1],[1],[1]])
print(v3)
v4=np.array([[1],[3],[1]])
print(v4)
left_hand_side= np.dot(v3.transpose(),v4)# Compute the dot product of v1 and v2
print('\n the result of lhs is:\n',left_hand_side)
norm_v3=np.linalg.norm(v3)# Compute the norms of v1 and v2
norm_v4=np.linalg.norm(v4)
print(norm_v3)
print(norm_v4)
right_hand_side=norm_v3*norm_v4# Compute the right-hand side of the Cauchy-Schwarz inequality
print('\n the right hand result:\n',right_hand_side)
# Check if the Cauchy-Schwarz inequality holds
cauchy_schwarz=left_hand_side<=right_hand_side
print('\nis the inequality holds?',cauchy_schwarz)


#EXERCISE:INFOSYS:
import numpy as np
A=np.array([[3,-1,2],
            [1,2,3],
            [4,1,1]])
print(A)
print('\nthe determinant:\n',np.linalg.det(A))
print('\nthe rank:\n',np.linalg.matrix_rank(A))
print('\n the inverse of    A:\n', np.linalg.inv(A))



#SOlVING LINEAR EQUATION:
import numpy as np
M=np.array([[1,3,-1],
            [2,5,4],
            [2,3,-1]])
print(M)
b=np.array([[4],[19],[7]])
x=np.linalg.solve(M,b)# Find the solution for the system of equations using the solve() method
print('\nthe value of x1:',x[0])
print('\nthe value of x2:',x[1])
print('\nthe value of x3:',x[2])

print(np.linalg.inv(M))#INVERSE OF MATRIX
print(np.linalg.pinv(M))#for generalised inverse

















