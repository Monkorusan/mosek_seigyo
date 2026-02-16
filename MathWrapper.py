import numpy as np
import mosek.fusion as mf
from typing import Any
import scipy.linalg

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
    """Bypasses Mosek API's incomprehensible documentation with familiar Python (or Numpy) syntaxes"""
    Expr: Any

    def __init__(self, obj: np.ndarray | Any):
        if isinstance(obj,np.ndarray):      #if obj is numpy type, convert to mosek
            self.Expr = mf.Matrix.dense(obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            self.Expr = mf.Matrix.dense(np.array(obj)) # Safety for lists passed as matrices
        elif isinstance(obj, MathWrapper):  #if obj is mw type, access its expr attribute
            self.Expr = obj.Expr
        else:                               #if already is mosek type
            self.Expr = obj 

    def __matmul__(self,other)-> 'MathWrapper':
        # other_expr = other.Expr if isinstance(other,MathWrapper) else other 
        # return MathWrapper(mf.Expr.mul(self.Expr,other_expr))
        if isinstance(other, MathWrapper):
            right = other.Expr
        elif isinstance(other, np.ndarray):
            right = mf.Matrix.dense(other) # CRITICAL FIX: Convert raw numpy to Mosek Matrix on the fly
        else:
            right = other 
        return MathWrapper(mf.Expr.mul(self.Expr, right))
    
    def __add__(self,other)-> 'MathWrapper':
        # other_expr = other.Expr if isinstance(other,MathWrapper) else other
        # return MathWrapper(mf.Expr.add(self.Expr, other_expr))
        if isinstance(other, MathWrapper):
            right = other.Expr
        elif isinstance(other, np.ndarray):
            right = mf.Matrix.dense(other)
        else:
            right = other
        return MathWrapper(mf.Expr.add(self.Expr, right))
    
    def __sub__(self, other)-> 'MathWrapper':
        # other_expr = other.Expr if isinstance(other,MathWrapper) else other
        # return MathWrapper(mf.Expr.sub(self.Expr, other_expr))
        if isinstance(other, MathWrapper):
            right = other.Expr
        elif isinstance(other, np.ndarray):
            right = mf.Matrix.dense(other)
        else:
            right = other
        return MathWrapper(mf.Expr.sub(self.Expr, right))

    
    def sqrt__(self) -> 'MathWrapper':
        """
        Element-wise square root for CONSTANT matrices only.
        """
        # 1. check if we can extract data (is it a constant?)
        if hasattr(self.Expr, "getDataAsArray"):
            # Extract raw data -> Numpy Sqrt -> New Wrapper
            raw_data = self.Expr.getDataAsArray() 
            return MathWrapper(np.sqrt(raw_data))
        
        # 2. If it's a Variable, we strictly forbid this.
        else:
            raise TypeError(
                "💢 Warning: Cannot call .sqrt() on a Variable directly in Mosek.\n"
                "You must use a Rotated Quadratic Cone constraint instead:\n"
                "M.constraint(Expr.vstack(y, 0.5, t), Domain.inRotatedQCone())"
            )

    @staticmethod
    def sqrt(M: 'MathWrapper') -> 'MathWrapper':
        """
        Robust Matrix Square Root.
        Extracts data from Mosek Matrix objects properly before calculation.
        """
        # 1. Unwrap safely
        if hasattr(M, "Expr"):
            raw = M.Expr
        else:
            raw = M

        # 2. EXTRACT DATA (The Fix)
        # If 'raw' is a Mosek Matrix object, we MUST ask it for the numbers first.
        if hasattr(raw, "getDataAsArray"): 
            # Mosek returns a flat list (1D array)
            flat_data = raw.getDataAsArray()
            # We must reshape it to 2D manually using its dimensions
            rows = raw.numRows()
            cols = raw.numColumns()
            data = np.array(flat_data).reshape((rows, cols))
        else:
            # Assume it's already a numpy array or list (Standard path)
            data = np.array(raw, dtype=float)
        
        # 3. Calculate SqrtM (Scipy)
        res = scipy.linalg.sqrtm(data)

        # 4. Clean up (Real part only + Ensure 2D)
        if np.iscomplexobj(res):
            res = res.real
        
        if res.ndim == 1:
            n = int(np.sqrt(res.size))
            res = res.reshape((n, n))
            
        return MathWrapper(res)

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

    def __lshift__(self,other)-> 'MathWrapper':
        """
        Syntax Sugar for LMI: returns A << B ,which is A ≺ B mathematically.
        A ≺ B is equivalent to B - A ≻ 0
        for more explanation, go read __rshift__ method's docstring.
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
        
    def trace(self) -> 'MathWrapper':
        """Calculates the trace of a matrix, could've been @property but 
        to avoid confusion for numpy user, trace here is intentionally kept as a method 
        rather than an attribute."""
        if hasattr(self.Expr, "diag"):
            return MathWrapper(mf.Expr.sum(self.Expr.diag()))
        else: #if no diag attribute found, we do it the vectorized way
            shape = self.Expr.getShape()
            n = shape[0]
            I = mf.Matrix.eye(n)
            return MathWrapper(mf.Expr.dot(self.Expr,I))
        

        
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
        if hasattr(self.Expr,"level"):
            raw = np.array(self.Expr.level(), dtype=float)
            if hasattr(self.Expr, "getShape"):
                shape = tuple(self.Expr.getShape())
                if len(shape) == 2:
                    return raw.reshape(shape)
                if len(shape) == 1:
                    return raw.reshape(shape[0],)
            return raw
        else:
            raise NotImplementedError("💢 warning: Can not get value of a non-variable expression")
    
    @staticmethod
    def ones(rows,cols=None)-> 'MathWrapper':
        """create an matrix  of size n filled with ones just like np.ones(n)"""
        if cols is None:
            cols = rows
        return MathWrapper(mf.Matrix.ones(rows,cols))
    
    @staticmethod
    def eye(n:int)-> 'MathWrapper':
        """create an identity matrix of size n just like np.ones(n)"""
        return MathWrapper(mf.Matrix.eye(n))
    
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
                raise TypeError("💢 ERROR, MathWrapper only support slicing for slice type and int type, use .pick() for lists")
        return MathWrapper(self.Expr.slice(startlist,endlist))
    
    @staticmethod
    def auto_bmat(rows_list:list[list[Any]])-> 'MathWrapper':
        """create block matrix where the 'None' part are assumed to be zero matrix automatically.
        Inspiration? MATLAB's robust_control_toolbox LMIterm function! 
        use example: 
        auto_bmat([[ A ,   B  , None],
                   [ C , None ,  D  ]]) 
        this will return a AB0C0D where the zero block automatically adjusts its dimension along with nonzero matrices!!
        """
        final_expr = None
        nrows = len(rows_list)
        ncols = len(rows_list[0])
        row_height:list[Any] = [None]*nrows #list of num of rows of each block matrix
        col_width :list[Any] = [None]*ncols #list of num of cols of each block matrix
        shape:Any=(0,0)
    
        for r in range(nrows):#scan2find dim
            for c in range(ncols):
                elem = rows_list[r][c] #block element
                if elem is not None:
                    raw_expr = elem.Expr if isinstance(elem, MathWrapper) else elem # unwrap if mosek object, else dont bother
                    if isinstance(raw_expr, mf.Expression):
                        shape = raw_expr.getShape()
                    elif isinstance(raw_expr,mf.Matrix):
                        shape = (raw_expr.numRows(),raw_expr.numColumns())
                    elif isinstance(raw_expr, np.ndarray):
                        shape = raw_expr.shape
                    else:
                        raise NotImplementedError(f"[Coming soon] at ({r},{c}), unimplemented data type {type(raw_expr)} detected (currently only support mf.Expr, mf.Matrix, np.array).") 

                    if row_height[r] is None: #if current is first block in current row,
                        row_height[r] = shape[0]
                    elif row_height[r] != shape[0]:
                        raise ValueError(f"row {r} has inconsistent height!,Expected {row_height[r]}, got {shape[0]} at col {c}.")
                    
                    if col_width[c] is None: #if current is first block in the current column,
                        col_width[c] = shape[1]
                    elif col_width[c] != shape[1]:
                        raise ValueError(f"col {c} has inconsistent width!,Expected {col_width[c]}, got {shape[1]} at row {r}.")
                    
        def to_expression(raw: Any) -> Any:
            if isinstance(raw, mf.Expression):
                return raw
            if isinstance(raw, mf.Matrix):
                return mf.Expr.constTerm(raw)
            if isinstance(raw, np.ndarray):
                return mf.Expr.constTerm(mf.Matrix.dense(raw))
            if isinstance(raw, (int, float)):
                return mf.Expr.constTerm(np.array([[float(raw)]]))
            return raw

        clean_rows = []
        for r in range(nrows): #add zero block matrix to parts with type 'None'
            current_row_expr = []
            for c in range(ncols):
                elem = rows_list[r][c]
                if elem is None:
                    if row_height[r] is None or col_width[c] is None:
                        raise ValueError(f"Block at ({r},{c}) is None, but entire row/col is empty. Cannot infer size.")
                    current_row_expr.append(mf.Expr.zeros([row_height[r],col_width[c]])) #append constant zero matrix object to list
                else:
                    raw = elem.Expr if isinstance(elem, MathWrapper) else elem
                    current_row_expr.append(to_expression(raw))
            clean_rows.append(mf.Expr.hstack([to_expression(item) for item in current_row_expr])) 

        final_expr = mf.Expr.vstack(clean_rows)
        return MathWrapper(final_expr) #stacking list vertically results in a matrix

    


