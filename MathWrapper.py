import numpy as np
import mosek.fusion as mf
from typing import Any

def block(rows):
    """
    imitate np.block() method, e.g. block([[A,B],[C,D]]) then return block matrix 
    """
    raw_rows = []
    for row in rows:
        raw_row = [item.Expr if isinstance(item,MathWrapper) else item for item in rows]
        raw_rows.append(mf.Expr.hstack(raw_row)) #stack horizontally then
    return MathWrapper(mf.Expr.vstack(raw_rows)) #stack vertically then get block matrix

class MathWrapper:
    """Bypasses Mosek API's incomprehensble documentation with familiar Python (or Numpy) syntaxes"""

    def __init__(self,Expr)->None:
        self.Expr = Expr

    def __matmul__(self,other)-> 'MathWrapper':
        other_expr = other.Expr if isinstance(other,MathWrapper) else other 
        return MathWrapper(mf.Expr.mul(self.Expr,other_expr))
    
    def __add__(self,other)-> 'MathWrapper':
        other_expr = other.Expr if isinstance(other,MathWrapper) else other
        return MathWrapper(mf.Expr.add(self.Expr, other_expr))
    
    def __sub__(self, other)-> 'MathWrapper':
        other_expr = other.Expr if isinstance(other,MathWrapper) else other
        return MathWrapper(mf.Expr.sub(self.Expr, other_expr))

    def __getattr__(self,name)->Any: 
        """not to be confused with __getattribute__ which modifies how returned value is accessed"""
        return getattr(self.Expr,name)
    
    def __neg__(self)-> 'MathWrapper':
        return MathWrapper(mf.Expr.neg(self.Expr))
    
    def __str__(self)->str:
        """modifies print() behavior: dont have to look at memory address"""
        return str(self.Expr)
    
    def __repr__(self)->str:
        """called by python3 console output or breakpoint()"""
        return f"MathWrapper({str(self.Expr)})"
    
    def __rshift__(self,other)-> 'MathWrapper':
        """
        Syntax Sugar for LMI:  
        A >> B  returns the expression (A - B) being a positive definite matrix.
        A friendly reminder: 
        In coding practice, expressing Positive Definiteness (≻) is not possible directly, 
        unless we replace it with the Positive Semidefinite operator (⪰).
        Mathematically, we bypass this by enforcing:
            (A - I) ⪰ 0   which is equivalent to   A ≻ 0
        use example: expressing A ≻ 0 can be done with A >> I
        """
        if isinstance(other,(int,float)) and other == 0:
            return self #handle P ≻ 0
        if not isinstance(other,MathWrapper):
            try:
                other = MathWrapper(other)
            except:
                print("💢 warning: can not wrap the object passed in MathWrapper.__rshift__")
        return self - other # reminder: we reuse MathWrapper.__sub__(self,other) here.

    @property
    def T(self)-> 'MathWrapper':
        """allowing access to a matrix's transpose with .T , handle both variable matrix and constant matrix"""
        if isinstance(self.Expr,mf.Matrix): #if constant matrix
            return MathWrapper(self.Expr.transpose())
        elif isinstance(self.Expr,mf.Expression): # if variable, sum, product etc.... 
            return MathWrapper(mf.Expr.transpose(self.Expr))
        else:
            raise TypeError(f"💢 warning, can not transpose object of type {type(self.Expr)}")
        
    @property
    def shape(self)->tuple[int,int]:
        """return dimension of object as a tuple of integer , e.g. (3,3)"""
        return tuple(self.Expr.getShape())
    
    @property
    def ndim(self)->int:
        """return dimension of square matrix-like object as an integer """
        return self.Expr.getND()
    
    def reshape(self, *args)-> 'MathWrapper':
        """handle both .reshape(int,int) and .reshape(list[int,int]) , same as numpy.reshape"""
        if len(args) == 1 and isinstance(args[0],(list,tuple)): #if arg is a list or tuple
            dims = args[0]
        else:
            dims = list(args) #listify the args
        return MathWrapper(self.Expr.reshape(dims))
    
    def flatten(self)-> 'MathWrapper':
        """flatten matrix to vector, similar to numpy.flatten"""
        return MathWrapper(self.Expr.flatten())
    
    def val(self)->np.ndarray:
        """return numerical value after solving, because __str__ doesn't do that.
        This method works universally now! regardless of matrix being dense or sparse then return as matrix array instead of flattened one."""
        try:
            temp = self.Expr.level()
        except AttributeError:
            temp = self.Expr.getDataAsarray()
        return np.array(temp).reshape(self.shape)
    
    @staticmethod
    def ones(rows,cols=None)-> 'MathWrapper':
        """create an identity matrix of size n just like np.eye(n)"""
        if cols is None:
            cols = rows
        return MathWrapper(mf.Matrix.ones(rows,cols))
    
    @staticmethod
    def zeros(rows,cols=None)-> 'MathWrapper':
        """create an matrix of zeros of size n just like np.zeros(n)"""
        if cols is None:
            cols = rows
        return MathWrapper(mf.Matrix.sparse(rows,cols))

    def __getitem__(self,key)-> 'MathWrapper':
        """allows Pythonic slicing, e.g. A[0:1,0:3] etc..."""
        shape = self.Expr.getShape()
        if not isinstance(key,tuple):
            key = (key,)
        startlist = []
        endlist = []
        for dim,k in enumerate(key): #loop thru key tuple for dimension checking
            # k:str , dim:int
            if isinstance(k,slice): # if k is a slice object (of some matrix)
                s = k.start if k.start is not None else 0
                e = k.stop if k.stop is not None else shape[dim]
                startlist.append(s)
                endlist.append(e)
            elif isinstance(k,int): #if it's not a slice but rather an element instead
                startlist.append(k)
                endlist.append(k+1)
            else:
                raise TypeError("ERROR, MathWrapper only support slicing for slice type and int type, use .pick() for lists")
        return MathWrapper(self.Expr.slice(startlist,endlist))

def main():
    with mf.Model("bla bla") as m:
        A = MathWrapper(m.variable("A", mf.Domain.inPSDCone(4)))
        A11 = A[0:2, 0:2]  
        A12 = A[0:2, 2:4]
        print(f"A11 = {A11}\n")
        print(f"A12 = {A12}\n")

if __name__ == "__main__":
    main()


# Works exactly like NumPy, but returns MOSEK Expressions.
# templist = [[1,2,3],
#             [4,5,6],
#             [7,8,9]]
# A = np.array(templist)
# # print(A[0:2,0:2])
# # print(A[0,2])
# print(A[[0]])
# print(A[0:1,:])
    


