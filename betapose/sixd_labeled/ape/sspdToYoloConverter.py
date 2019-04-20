
import glob, os

for file in glob.glob("*.txt"):
    f = open(file, "r")
    line = f.read()

    lineVals = line.split()

    newLine = lineVals[0] + ' ' + lineVals[1] + ' ' + lineVals[2] + ' ' + lineVals[19] + ' ' + lineVals[20]

    with open('./converted/' + file, 'w') as file:
        file.write(newLine)

