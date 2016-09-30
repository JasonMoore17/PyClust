from __future__ import print_function
from setuptools import setup

import numpy
import sys
sys.setrecursionlimit(5000)

# generic options
options = {
    'name': 'PyClust',
    'version'   : '0.0.1',
    'description'   : 'A python/pyside spike sorting tool for tetrode data',
    'author'        : 'Bernard Willers',
    'data_files'    : [('sample', ['sample/Sample1.ntt', 'sample/Sample2.ntt',
                            'sample/Sample3.ntt', 'sample/Sample4.ntt',
                            'sample/Sample5.ntt', 'sample/TT22.spike'])],
    'url'           : 'https://github.com/bwillers/pyclust',
}

# windows specific
if len(sys.argv) >= 2 and sys.argv[1] == 'py2exe':
    try:
        import py2exe
    except ImportError:
        print('Could not import py2exe, windows build aborted')
        sys.exit(0)
    import matplotlib
    options['options'] = {
        'py2exe': {
            'includes': ['scipy.io.matlab.streams','scipy.special._ufuncs_cxx','scipy','scipy.linalg.*','scipy.sparse.csgraph._validation','sklearn.utils.lgamma','sklearn.neighbors.typedefs','sklearn.utils.sparsetools._graph_validation','sklearn.utils.weight_vector'],
            'excludes': ['_wxagg', '_gtkagg', '_tkagg']
        }
    }

    data_files = matplotlib.get_py2exe_datafiles()

    # something weird with enthought i need to include these manually
    #data_files.append(r'C:\Python27\Scripts\mk2_core.dll')
   # data_files.append(r'C:\Python27\Scripts\mk2_mc3.dll')
   # data_files.append(r'C:\Python27\Scripts\mk2iomp5md.dll')
   # data_files.append(r'scipy.sparse.csgraph._validation')
   # data_files.append(r'scipy.special._ufuncs_cxx')

    options['data_files'] = options['data_files'] + data_files

    options['console'] = ['pyclust.py']

# mac specific
if len(sys.argv) >= 2 and sys.argv[1] == 'py2app':
    try:
        import py2app
    except ImportError:
        print('Could not import py2app, mac build aborted')
        sys.exit(0)
    # mac options
    options['app'] = ['pyclust.py']
    options['options'] = {
        'py2app': {
            'argv_emulation': True,
            #  I dont use it, but it tries to build it and fails
            'excludes': ['sympy', 'PyQt4', 'sip', 'virtualenv',
                         'email', 'pyzmq', 'IPython', 'jinja',
                         'pygments', 'tornado', 'weave'],
            'includes':['sklearn.metrics.cluster.expected_mutual_info_fast',
                        'sklearn.utils.lgamma',
                        'sklearn.utils.weight_vector']
        }
    }

print(options)

# run setup
setup(**options)


