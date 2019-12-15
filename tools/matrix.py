#! /usr/sys/env python 3



def shape(A):
	num_rows = len(A)
	num_cols = len(A[0]) if A else 0
	return num_rows,num_cols

def get_row(A,i):
	return A[i]

def get_column(A,i):
	return [A_i[i] for A_i in A]

def make_matrix(num_rows,num_cols,entry_fn):
	return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def is_diagonal(i,j):
	return 1 if i==j else 0


