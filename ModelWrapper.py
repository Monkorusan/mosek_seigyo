import numpy as np
import mosek.fusion as mf
from MathWrapper import MathWrapper as mw
from typing import Any

class ModelWrapper(mf.Model):
    """
    A wrapper for mosek.fusion.Model: auto-wraps and unwraps expressions
    without this class, accessing the expression P>>I  requires calling (P>>I).Expr 
    """

    # def variable(self, *args)->mw: 
    #     """overwrite mosek.fusion.Model.variable method directly.
    #     a friendly reminder: overwritting parent method in child class requires calling super() method"""
    #     temp_var = super().variable(*args) 
    #     return mw(temp_var)

    def variable(self, *args) -> mw:
            """
            Smart Wrapper:
            1. Detects if a tuple shape (r, c) was passed.
            2. Unpacks it so Mosek receives integers.
            """
            new_args = []
            for arg in args:
                # If we find a tuple shape like (2, 2), convert it to a List [2, 2]
                # Mosek accepts Lists for shapes, but hates Tuples.
                if isinstance(arg, tuple):
                    new_args.append(list(arg))
                else:
                    new_args.append(arg)
            
            # Pass the cleaned arguments to Mosek
            temp_var = super().variable(*new_args)
            return mw(temp_var)
    
    def constraint(self,*args):
        """overwrite mosek.fusion.Model.constraint method directly.
           remember that MOSEK constraint object require declaring demain , usually pos-semi-definite."""
        arg_list = []
        for arg in args:
            if hasattr(arg, "Expr"): 
                arg_list.append(arg.Expr)
            else:
                arg_list.append(arg)

        if len(arg_list) == 1:
            expr = arg_list[0]
            if hasattr(expr, "getShape"):
                shape = expr.getShape()
                if len(shape) == 2 and shape[0] == shape[1]:
                    return super().constraint(expr, mf.Domain.inPSDCone(shape[0]))
            raise TypeError("Single-argument constraint requires a square matrix expression for PSD domain inference.")

        return super().constraint(*arg_list)
    
    def const_matrix(self, data:Any) -> mw:
        """create a matrix object while bypassing calling mw in the solve method in your code. just by replacing
        <with mf.Model(self.name) as M:> with <with ModelWrapper(self.name) as M:> 
        NEW UPDATE:
        # Force conversion to a Numpy Array of Floats immediately.
        # This prevents "lists" or "object arrays" from sneaking in."""
        clean_data = np.array(data, dtype=float)
        return mw(clean_data)
    
    def objective(self, name, sense, expr):
        """ Overwrites M.objective to auto-unwrap MathWrapper objects.
            without this method overriding, unwrapping object at def solve with .Expr is strictly required. """
        raw_expr = expr.Expr if hasattr(expr,"Expr") else expr
        return super().objective(name,sense,raw_expr)
