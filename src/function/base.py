import os
from src.function.utils import create_function_docs
from src.function.example import *

def find_func(func_name):
    if func_name in globals():
        myfunc = globals()[func_name]  # Lookup the function in global scope
        if callable(myfunc):          # Check if it is callable
            return myfunc                 # Call the function
        else:
            print(f"{func_name} is not callable!")
    else:
        print(f"Function {func_name} not found!")


class BaseFunctionList:
    def __init__(self, functions=None):
        ## TODO: functions to path
        if functions is None:
            functions = ["book_accommodation","book_flight","book_rental_car"]
        self.func_list = [find_func(func_name) for func_name in functions]
        self.func_doc = create_function_docs(self.func_list)

