import os
import subprocess

path = 'bigIcons'
files = os.listdir(path)
for fname in files:
    print fname
    (main, ext) = fname.split('.')
    subprocess.call (['convert', '-fx', 'G', '-resize', '80x', path+'/'+fname, 'icons/'+main+'.jpg'])


