import glob
ff = glob.glob("drawmodules/*.py")

import re
mods = re.compile('.*/([^_].*)\.py')

for l in ff:
    m = mods.match(l)
    if m is not None:
        print("imported from testmodules", m.group(1))
        exec("from . import "+ m.group(1))
        import importlib
        exec('importlib.reload('+m.group(1)+')')

    # l2 = l.split('/')
    # l3 = l2[1].split('.')
    # if l3[0][0] != '_':
    #     print(l3[0])
    #     exec("from . import "+l3[0])


print('testmodules loaded')
