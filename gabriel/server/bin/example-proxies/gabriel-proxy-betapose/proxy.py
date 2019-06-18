#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#
#   Author: Kiryong Ha <krha@cmu.edu>
#           Zhuo Chen <zhuoc@cs.cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import cv2
import json
import multiprocessing
import numpy as np
import os
import pprint
import queue
import struct
import sys
import time
import wave
import sspd.evaluate_single_image

dir_file = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_file, "../../.."))
import gabriel3
import gabriel3.proxy
LOG = gabriel3.logging.getLogger(__name__)


def raw2cv_image(raw_data):
    img_array = np.asarray(bytearray(raw_data), dtype=np.int8)
    cv_image = cv2.imdecode(img_array, -1)
    return cv_image

def fix_image_ratio(img, is_resize = True, resize_method = "max", resize_max = -1, resize_scale = 1, save_image = False):
    '''
    Display image at appropriate size. There are two ways to specify the size:
    1. If resize_max is greater than zero, the longer edge (either width or height) of the image is set to this value
    2. If resize_scale is greater than zero, the image is scaled by this factor
    '''
    if is_resize:
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        if resize_max > 0:
            if height > width:
                img_display = cv2.resize(img, (int(resize_max * width / height), resize_max),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                img_display = cv2.resize(img, (resize_max, int(resize_max * height / width)),
                                         interpolation=cv2.INTER_NEAREST)
        elif resize_scale > 0:
            img_display = cv2.resize(img, (int(width * resize_scale), int(height * resize_scale)),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            print("Unexpected parameter in image display. About to exit...")
            sys.exit()
    else:
        img_display = img
    return img_display

def display_image(display_name, img_display, wait_time = -1):
    cv2.imshow(display_name, img_display)
    cv2.waitKey(wait_time)

class DummyVideoApp(gabriel3.proxy.CognitiveProcessThread):
    def handle(self, header, data):
        # PERFORM Cognitive Assistance Processing
        header['status'] = "success"
        print("processing: ")
        print("%s\n" % header)
        img = fix_image_ratio(raw2cv_image(data), resize_max = 640)
        eval_res = sspd.evaluate_single_image.evaluate_img(img)
        display_image('input', eval_res[0], wait_time = 1)
        return json.dumps({'center': str(eval_res[1]), 'bb': str(eval_res[2])})


if __name__ == "__main__":
    import sys
    if (len(sys.argv) == 5):
        datacfg_file = sys.argv[1]  # data file
        cfgfile_file = sys.argv[2]  # yolo network file
        weightfile_file = sys.argv[3]  # weightd file
        sspd.evaluate_single_image.initialize_network(datacfg_file, cfgfile_file, weightfile_file)

        #settings = gabriel3.util.process_command_line(sys.argv[1:])
        settings = sys.argv[4]
        ip_addr, port = gabriel3.network.get_registry_server_address(settings)
        service_list = gabriel3.network.get_service_list(ip_addr, port)
        print("Gabriel Server :")
        print(pprint.pformat(service_list))

        video_ip = service_list.get(gabriel3.ServiceMeta.VIDEO_TCP_STREAMING_IP)
        video_port = service_list.get(gabriel3.ServiceMeta.VIDEO_TCP_STREAMING_PORT)
        acc_ip = service_list.get(gabriel3.ServiceMeta.ACC_TCP_STREAMING_IP)
        acc_port = service_list.get(gabriel3.ServiceMeta.ACC_TCP_STREAMING_PORT)
        audio_ip = service_list.get(gabriel3.ServiceMeta.AUDIO_TCP_STREAMING_IP)
        audio_port = service_list.get(gabriel3.ServiceMeta.AUDIO_TCP_STREAMING_PORT)
        ucomm_ip = service_list.get(gabriel3.ServiceMeta.UCOMM_SERVER_IP)
        ucomm_port = service_list.get(gabriel3.ServiceMeta.UCOMM_SERVER_PORT)

        # this queue is shared by multiple sensor processing threads
        result_queue = multiprocessing.Queue()

        # image receiving and processing
        image_queue = queue.Queue(gabriel3.Const.APP_LEVEL_TOKEN_SIZE)
        print("TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel3.Const.APP_LEVEL_TOKEN_SIZE)
        video_streaming = gabriel3.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
        video_streaming.start()
        video_streaming.isDaemon = True

        video_app = DummyVideoApp(image_queue, result_queue, engine_id="Dummy_video")
        video_app.start()
        video_app.isDaemon = True

        # result pub/sub
        result_pub = gabriel3.proxy.ResultPublishClient((ucomm_ip, ucomm_port), result_queue)
        result_pub.start()
        result_pub.isDaemon = True

        try:
            while True:
                time.sleep(1)
        except Exception as e:
            pass
        except KeyboardInterrupt as e:
            LOG.info("user exits\n")
        finally:
            if video_streaming is not None:
                video_streaming.terminate()
            if video_app is not None:
                video_app.terminate()
            result_pub.terminate()

    else:
        print('Usage:')
        print('python evaluateSingleImage.py datacfg cfgfile weightfile garbiel-control-server-address')

