## 79 ########################################################################

# python3 utilz2/dev/project.py --src tac --tag with_net
from utilz2 import *
if __name__ == '__main__':
    import sys
    print("Argument List:", str(sys.argv))
    s=select_from_list(['findideal','gen0'])
    if s=='findideal':
        from .findideal import *
    elif s=='gen0':
        from .gen0 import *
    else:
        assert False

#EOF