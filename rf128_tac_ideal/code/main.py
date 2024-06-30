## 79 ########################################################################

# python3 utilz2/dev/project.py --src tac --tag with_net
from utilz2 import *
if __name__ == '__main__':
    import sys
    print("Argument List:", str(sys.argv))
    s=select_from_list(['gen0'])#['findideal','gen0','gen1','test0'])
    if s=='findideal':
        from .findideal import *
    elif s=='gen0':
        from .gen0 import *
    elif s=='gen1':
        from .gen1 import *
    elif s=='test0':
        from .test0 import *
    else:
        assert False

#EOF