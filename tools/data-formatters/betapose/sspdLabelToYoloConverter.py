import glob, os, shutil

if not os.path.exists('./converted'):
    os.makedirs('./converted')

os.chdir('./labels')
for file in glob.glob("*.txt"):
    f = open(file, "r")
    line = f.read()

    lineVals = line.split()
    if (len(lineVals) > 19):
        newLine = lineVals[0] + ' ' + lineVals[1] + ' ' + lineVals[2] + ' ' + lineVals[19] + ' ' + lineVals[20]
    else:
        newLine = ' '
    with open('../converted/' + file, 'w') as file:
        file.write(newLine)
os.chdir('../')

# delete all files in labels
shutil.rmtree('./labels')

# move converted to labels
os.rename('./converted', './labels')

# fix train and test
cwd = os.getcwd()
with open('./train_new.txt', 'w') as file:
    f = open('./train.txt', "r")
    for x in f:
        file.write(x.replace('sspdFormat', cwd).replace('g ', 'g'))
    f.close()

with open('./test_new.txt', 'w') as file:
    f = open('./test.txt', "r")
    for x in f:
        file.write(x.replace('sspdFormat', cwd).replace('g ', 'g'))
    f.close()

os.remove('./train.txt')
os.remove('./test.txt')
os.rename('./train_new.txt', './train.txt')
os.rename('./test_new.txt', './test.txt')
