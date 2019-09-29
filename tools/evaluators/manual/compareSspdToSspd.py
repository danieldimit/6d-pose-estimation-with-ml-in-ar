import os
import time
import scipy.io
import warnings
warnings.filterwarnings("ignore")

from utils import *
from MeshPly import MeshPly

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def valid():
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    meshname     = 'kuka.ply'

    # Parameters
    testtime        = True
    use_cuda        = True
    num_classes     = 1
    testing_samples = 0.0
    eps             = 1e-5
    notpredicted    = 0 
    conf_thresh     = 0.1
    nms_thresh      = 0.4
    match_thresh    = 0.5

    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    errs_corner2D       = []
    preds_trans         = []
    preds_rot           = []
    preds_corners2D     = []
    gts_trans           = []
    gts_rot             = []
    gts_corners2D       = []

    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D     = get_3D_corners(vertices)
    # diam          = calc_pts_diameter(np.array(mesh.vertices))
    diam          = 0.6536639

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()

    # Iterate through test batches (Batch size for test data is 1)

    manuals=[]
    truths=[]
     # Iterate through each ground-truth object
    for file in sorted(os.listdir("./manual/")):
        f = open("./manual/" + file, "r")
        manuals.append(f.read().split(" "))
    for file in sorted(os.listdir("./gt/")):
        f = open("./gt/" + file, "r")
        truths.append(f.read().split(" "))
    count = 0
    for k in range(len(manuals)):
        box_gt        = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], truths[k][5], truths[k][6], 
                        truths[k][7], truths[k][8], truths[k][9], truths[k][10], truths[k][11], truths[k][12], 
                        truths[k][13], truths[k][14], truths[k][15], truths[k][16], truths[k][17], truths[k][18], 1.0, 1.0, truths[k][0]]

        box_pr        = [manuals[k][1], manuals[k][2], manuals[k][3], manuals[k][4], manuals[k][5], manuals[k][6], 
                        manuals[k][7], manuals[k][8], manuals[k][9], manuals[k][10], manuals[k][11], manuals[k][12], 
                        manuals[k][13], manuals[k][14], manuals[k][15], manuals[k][16], manuals[k][17], manuals[k][18], 1.0, 1.0, manuals[k][0]]

        # Denormalize the corner predictions 
        corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
        corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
        corners2D_gt[:, 0] = corners2D_gt[:, 0] * 640
        corners2D_gt[:, 1] = corners2D_gt[:, 1] * 480               
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

        preds_corners2D.append(corners2D_pr)
        gts_corners2D.append(corners2D_gt)

        # Compute corner prediction error
        corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
        corner_dist = np.mean(corner_norm)
        errs_corner2D.append(corner_dist)
        
        # Compute [R|t] by pnp
        R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))
   
        # Compute translation error
        trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
        errs_trans.append(trans_dist)
        
        # Compute angle error
        angle_dist   = calcAngularDistance(R_gt, R_pr)
        errs_angle.append(angle_dist)
        
        # Compute pixel error
        Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
        Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
        proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration)
        proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
        norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
        pixel_dist   = np.mean(norm)
        errs_2d.append(pixel_dist)

        # Compute 3D distances
        transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
        transform_3d_pred = compute_transformation(vertices, Rt_pr)  
        norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
        vertex_dist       = np.mean(norm3d)    
        errs_3d.append(vertex_dist)  

        # Sum errors
        testing_error_trans  += trans_dist
        testing_error_angle  += angle_dist
        testing_error_pixel  += pixel_dist
        testing_samples      += 1
        count = count + 1

    # Compute 2D projection error, 6D pose error, 5cm5degree error
    px_threshold = 5
    acc         = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc5cm5deg  = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    acc3d10     = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    acc5cm5deg  = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    corner_acc  = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)


    # Print test statistics
    logging('Results of {}'.format('kuka'))
    logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
    logging('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )

if __name__ == '__main__':
    import sys
    # the model ply must be in m
    # the betapose t must be in m
    # the kpd model ply must be in mm
    valid()
    
