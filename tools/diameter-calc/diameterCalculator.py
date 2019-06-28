import math
import numpy as np
import warnings
import threading
import queue
warnings.filterwarnings("ignore")
from MeshPly import MeshPly
import multiprocessing



def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))

def calc_pts_batch(pts, t, batch_stop_i, que):
    diameter = -1
    len = pts.shape[0]
    print('thread:' + str(t) + "len: "+ str(len))
    old_percent = 0;
    for pt_id in range(len):
        if (pt_id > batch_stop_i):
            break
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        new_percent = int(pt_id / batch_stop_i * 10)
        if (old_percent < new_percent):
            old_percent = new_percent
            print('{} -- {}%'.format(t, old_percent * 10))
        if max_dist > diameter:
            diameter = max_dist
    print('Finished {} -- {:.5f}'.format(t, diameter))
    que.put(diameter)

def calc_pts_diameter(pts):
    len = pts.shape[0]
    n_threads = max(multiprocessing.cpu_count() - 2, 1)
    batch_size = int(len / (n_threads * 2) )
    que = queue.Queue()
    thread_pool = list()
    for t in range(n_threads):
        batch_start = t * batch_size
        batch_stop = min((t + 1) * batch_size, len)
        x = threading.Thread(target=calc_pts_batch, args=(pts[batch_start:, :],t,batch_stop-batch_start,que))
        thread_pool.append(x)
        x.start()

    diameter = -1
    for t in range(n_threads):
        thread_pool[t].join()

    while not que.empty():
        max_dist = que.get()
        if max_dist > diameter:
            diameter = max_dist

    return max_dist




def valid(ply):
    with open(ply) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

        skip = 0
        foundVertexEl = False
        foundEndOfHead = False
        lineVals = []
        linesToScan = 0

        while (not foundVertexEl or not foundEndOfHead):
            lineVals = content[skip].split()
            if (lineVals[0] == 'end_header'):
                foundEndOfHead = True
            if (lineVals[0] == 'element'):
                if (lineVals[1] == 'vertex'):
                    linesToScan = int(lineVals[2])
                    foundVertexEl = True
            skip += 1
        content = content[skip:linesToScan + skip]
        copy = [];
        for line in content:
            copy.append(splitAndCastToFloat(line))
        vertices = np.matrix(np.array(copy))
        mins = vertices.min(0)
        maxs = vertices.max(0)
        minsMaxs = np.array([[mins.item(0), mins.item(1), mins.item(2)], [maxs.item(0), maxs.item(1), maxs.item(2)]]).T


    print('Mins, Max: ')
    print(minsMaxs)

    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(ply)
    diam          = calc_pts_diameter(np.array(mesh.vertices))

    print('Diameter: ')
    print(diam)
    models_info = open('models_info.yml', "w+")
    models_info.write('1: {{ diameter: {:.5f}, mix_x: {:.5f}, min_y: {:.5f}, min_z: {:.5f}, size_x: {:.5f}, size_y: {:.5f}, size_z: {:.5f}}}'.format(diam, mins.item(0), mins.item(1), mins.item(2), abs(maxs.item(0)) + abs(mins.item(0)), abs(maxs.item(1)+abs(mins.item(1))) , abs(maxs.item(2))+ abs(maxs.item(2))))
    models_info.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        ply = sys.argv[1]
        valid(ply)
    else:
        print('Usage:')
        print(' python diameterCalculator.py psp.ply')
