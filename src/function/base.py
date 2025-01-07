import os
from src.function.utils import create_function_docs
from src.function.example import *

class BaseFunctionList:
    def __init__(self, flist=None):
        ## TODO: flist to path
        if flist is None:
            self.func_list = [book_accommodation,book_flight,book_rental_car]
        elif os.path.isfile(flist):
            pass
        else:
            self.func_list = flist
        self.func_doc = create_function_docs(self.func_list)

