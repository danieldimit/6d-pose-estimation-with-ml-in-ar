import math
import numpy as np
import warnings
import threading
import queue
warnings.filterwarnings("ignore")
from MeshPly import MeshPly
import multiprocessing


def substitute_numbers(line, vals):
    line = line.split()
    line[0] = "{:.5f}".format(vals[0])
    line[1] = "{:.5f}".format(vals[1])
    line[2] = "{:.5f}".format(vals[2])
    return ' '.join(line) + ' \n'





def valid(ply):
    mesh = MeshPly(ply)
    mins = np.array(mesh.vertices).min(0)
    maxs = np.array(mesh.vertices).max(0)
    minsMaxs = np.array([[mins.item(0), mins.item(1), mins.item(2)], [maxs.item(0), maxs.item(1), maxs.item(2)]]).T
    diffs = (maxs + mins) / 2
    mesh.vertices = (np.array(mesh.vertices) - diffs)

    with open(ply) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

        skip = 0
        foundVertexEl = False
        foundEndOfHead = False
        lineVals = []
        linesToScan = 0
        f = open("result.ply","w+")

        while (not foundVertexEl or not foundEndOfHead):
            lineVals = content[skip].split()
            f.write(content[skip] + '\n')
            if (lineVals[0] == 'end_header'):
                foundEndOfHead = True
            if (lineVals[0] == 'element'):
                if (lineVals[1] == 'vertex'):
                    linesToScan = int(lineVals[2])
                    foundVertexEl = True
            skip += 1
        copy_content = content[linesToScan + skip:]
        content = content[skip:linesToScan + skip]
        copy = [];
        for i, line in enumerate(content):
            f.write(substitute_numbers(line, mesh.vertices[i]))

        for line in copy_content:
            f.write(line + ' \n')
        
        f.close()


    

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        ply = sys.argv[1]
        valid(ply)
    else:
        print('Usage:')
        print('python diameterCalculator.py psp.ply')
