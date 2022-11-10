import re
from bs4 import BeautifulSoup

#namePattern = re.Pattern("&nbsp;.+")

def splittUp(line):
    return line.split("</span>")

lines = []
with open ("playersHTML.txt", "r") as file:
    hasLine = True
    while hasLine:
        try:
            line = file.readline()
            if line:
                if line.startswith('<span class="hidedesktop displaymobile">'):
                    lines.append(line)
            else:
                hasLine = False
        except:
            print("Unable to decode line")
print(len(lines))
print(lines[0])
splitted = splittUp(lines[0])
print("__________________________")
for line in splitted:
    print(line)

