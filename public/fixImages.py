import HTMLParser
import Image
import sys
import re

if len(sys.argv) != 2:
    print "Usage: " + sys.argv[0] + " <html file>"
    sys.exit(1)

htmlFileName = sys.argv[1]

htmlFileIn = open(htmlFileName, 'r')
htmlFileOut = open(htmlFileName + '.new', 'w')

pattern = re.compile ("<img src=\"([^\"]*)\" />")

for line in htmlFileIn:
    m = pattern.search (line)
    if m:
        imName = m.group(1)
        im = Image.open(imName)
        sz = im.size
        newVal = "<img src=\"" + imName + "\" width=\"" + str(sz[0]) + "\" height=\"" + str(sz[1]) + "\" />"
        line = pattern.sub (newVal, line)
    htmlFileOut.write(line)

htmlFileIn.close()
htmlFileOut.close()


