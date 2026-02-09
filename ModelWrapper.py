import numpy as np
import mosek.fusion as mf
from MathWrapper import MathWrapper as mw

class ModelWrapper(mf.Model):
    """
    A wrapper for mosek.fusion.Model: auto-wraps and unwraps expressions
    without this class, accessing the expression P>>I  requires calling (P>>I).Expr 
    """

    def variable(self, *args)->mw: 
        """overwrite mosek.fusion.Model.variable method directly.
        a friendly reminder: overwritting parent method in child class requires calling super() method"""
        temp_var = super().variable(*args) 
        return mw(temp_var)
    
    def constraint(self,*args):
        """overwrite mosek.fusion.Model.variable method directly.
           remember that MOSEK constraint object require declaring demain , usually pos-semi-definite."""
        arg_list = []
        for arg in arg_list:
            if isinstance(arg,mw): #check if expression is passed as a wrapper of class MathWrapper
                arg_list.append(arg.Expr)
            else:
                arg_list.append(arg)
        return super().constraint(*arg_list)
    