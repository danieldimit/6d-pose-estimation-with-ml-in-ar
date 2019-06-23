import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from .KPD.src.utils.img import load_image, cropBox, im_to_torch
from .opt import opt
from .yolo.preprocess import prep_image, prep_frame, inp_to_image
from .pPose_nms import pose_nms, write_json
from .KPD.src.utils.eval import getPrediction
from .yolo.util import write_results, dynamic_write_results
from .yolo.darknet import Darknet
from tqdm import tqdm
import cv2
import json
import numpy as np
import sys
import time
from .utils.utils import pnp
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread
from .yolo.preprocess import letterbox_image
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue

# if opt.vis_fast:
#     from fn import vis_frame_fast as vis_frame
# else:
#     from fn import vis_frame
from IPython import embed # for debugging

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

class ImageLoader:
    def __init__(self, queueSize=1, reso=608):
        self.stopped = False
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Resize(size=(reso, reso), interpolation=3),
            transforms.ToTensor()
        ])
        if opt.sp:  # torch 0.4.1 can only use sp
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            p = Thread(target=self.getitem_yolo, args=())
        else:
            p = mp.Process(target=self.getitem_yolo, args=())
        p.daemon = True
        p.start()
        return self

    def getitem_yolo(self):
        while (not self.stopped):
            time.sleep(2)

    def getitem(self):
        # called in det_loader
        return self.Q.get()

    def putitem(self, img, frame):
        im_dim_list = []
        orig_img_list = []
        frame = [].append(frame)

        inp_dim = int(opt.inp_dim)
        img_k, orig_img, im_dim = prep_image(img, inp_dim)

        im_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_pil)

        img_k = self.transform(im_pil).unsqueeze(0)
        img = []
        img.append(img_k)
        im_dim_list.append(im_dim)
        orig_img_list.append(orig_img)

        with torch.no_grad():
            img = torch.cat(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if (self.Q.full()):
            self.Q.get()

        # img is Tensor of Tensors, originial image is cv2 image
        self.Q.put((img, orig_img_list, frame, im_dim_list))

    def length(self):
        return len(self.imglist)

    def len(self):
        return self.Q.qsize()



class DetectionLoader:
    def __init__(self, image_loader, cfg_path, weights_path, batchSize=1, queueSize=2):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet(cfg_path, reso=int(opt.inp_dim))
        self.det_model.load_weights(weights_path)
        print("Loading YOLO cfg from", cfg_path)
        print("Loading YOLO weights from", weights_path)
        self.det_model.eval()
        self.det_model.net_info['height'] = opt.inp_dim #input_dimension
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.image_loader = image_loader
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        while(not self.stopped):
            img, orig_img, frame, im_dim_list = self.image_loader.getitem()
            if img is None:
                continue
            with torch.no_grad():
                img = img.cuda()
                # Critical, use yolo to do object detection here!
                prediction = self.det_model(img)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence, opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    if self.Q.full():
                        self.Q.get()
                    self.Q.put((orig_img, frame, None, None, None, None, None))
                    continue
                dets = dets.cpu()

                # Scale for SIXD dataset

                reso = self.det_inp_dim
                im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                w, h = im_dim_list[:,0], im_dim_list[:,1]
                w_ratio = w / reso
                h_ratio = h / reso
                boxes = dets[:, 1:5]
                boxes[:,0] = boxes[:,0] * w_ratio
                boxes[:,1] = boxes[:,1] * h_ratio
                boxes[:,2] = boxes[:,2] * w_ratio
                boxes[:,3] = boxes[:,3] * h_ratio
                scores = dets[:, 5:6]
                
                # im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
                # scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # # coordinate transfer
                # dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                # dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                # dets[:, 1:5] /= scaling_factor
                # for j in range(dets.shape[0]):
                #     dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                #     dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                # boxes = dets[:, 1:5]
                # scores = dets[:, 5:6]

            # img.save(im_name[0].replace('rgb', 'results'))

            boxes_k = boxes
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                if self.Q.full():
                    self.Q.get()
                self.Q.put((orig_img, frame, None, None, None, None, None))
                continue
            inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)
            if self.Q.full():
                time.sleep(2)
            self.Q.put((orig_img, frame, boxes_k, scores, inps, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor:
    def __init__(self, detectionLoader, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        self.stopped = False

        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = pQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        while(not self.stopped):
            with torch.no_grad():
                (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()
                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None, None))
                    return
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None))
                    continue
                inp = im_to_torch(cv2.cvtColor(orig_img[0], cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DataWriter:
    def __init__(self, cam_K, left_number, kp_model_vertices, save_video=False,
                savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
                queueSize=1024):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        self.outQ = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')
        self.kp_3d = kp_model_vertices
        self.cam_K = cam_K
        self.left_number = left_number

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                if boxes is None:
                    if opt.save_img or opt.save_video or opt.vis:
                        img = orig_img
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
                else:
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    
                    preds_hm, preds_img, preds_scores = getPrediction(
                        hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

                    result = pose_nms(boxes, scores, preds_img, preds_scores)
                    result = {
                        'imgname': im_name,
                        'result': result
                    } # append imgname here.
                    # result here includes imgname, bbox, kps, kp_score, proposal_score
                    # Critical, run pnp algorithm here to get 6d pose.
                    # embed()
                    KP_REMAIN = self.left_number
                    if result['result']:
                        kp_score = np.array(result['result'][0]['kp_score'][:,0])
                        kp_2d = np.array(result['result'][0]['keypoints'])
                        kp_3d = np.array(self.kp_3d)
                        while(len(kp_2d)>KP_REMAIN):
                            delidx = np.argmin(kp_score, axis=0)
                            kp_score = np.delete(kp_score,delidx)
                            kp_2d = np.delete(kp_2d,delidx, axis=0)
                            kp_3d = np.delete(kp_3d,delidx, axis=0)
                        # embed()
                        R, t = pnp(kp_3d, kp_2d, self.cam_K)
                        result.update({'cam_R':R, 'cam_t':t})
                    else:
                        result.update({'cam_R':[], 'cam_t':[]})
                    self.final_result.append(result)
                    self.outQ.put(result)
                    # if opt.save_img or opt.save_video or opt.vis:
                    #     img = vis_frame(orig_img, result)
                    #     if opt.vis:
                    #         cv2.imshow("AlphaPose Demo", img)
                    #         cv2.waitKey(30)
                    #     if opt.save_img:
                    #         cv2.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                    #     if opt.save_video:
                    #         self.stream.write(img)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def read(self):
        return self.outQ.get()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name): # using update
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()

class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders # used in training
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        # Change number of keypoints here!
        self.nJoints_coco = 50
        self.nJoints_mpii = 50 #16
        self.nJoints = 50 #33

        self.accIdxs = tuple([i for i in range(1, 50 + 1)])
        self.flipRef = ()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img, upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
