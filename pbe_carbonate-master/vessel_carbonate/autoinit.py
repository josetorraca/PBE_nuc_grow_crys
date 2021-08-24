#autoinit.py
from inspect import getfullargspec
class AutoInit(type):
      def __new__(meta, classname, supers, classdict):
        classdict['__init__'] = autoInitDecorator(classdict['__init__'])
        return type.__new__(meta, classname, supers, classdict)

def autoInitDecorator (toDecoreFun):
    def wrapper(*args):
        argsnames = getfullargspec(toDecoreFun)[0]

        argsvalues = [x for x in args[1:]]

        # 'self' -> the reference to the instance
        objref = args[0]

        # setting the attribute with the corrisponding values to the instance
        # note I am skipping the 'self' reference
        for x in argsnames[1:]:
            objref.__setattr__(x,argsvalues.pop(0))

    return wrapper
