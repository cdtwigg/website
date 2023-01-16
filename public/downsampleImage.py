import sys
import subprocess
import os

for file in sys.argv[1:]:
    newFile = file + '.new'
    os.system ('jpegtran -optimize ' + file + ' > ' + newFile)
    os.rename (file, file + '.old')
    os.rename (newFile, file)



