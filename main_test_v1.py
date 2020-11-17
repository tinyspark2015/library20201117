import cv2
import time
import datetime
import os

import numpy as np
from pifpaf_slim_v3 import predict as pose_pred

import matplotlib.path as mplPath

import imageio
from PIL import Image,ImageFont,ImageDraw

'''Global Setting'''

# first column is main stream, second column is sub
rtsp_postfix = {
    "Dahua": ["/cam/realmonitor?channel=1&subtype=0", "/cam/realmonitor?channel=1&subtype=1"],
    "HIKvision": ["/h264/ch1/main/av_stream", "/h264/ch1/sub/av_stream"]
}

# 192.168.88.2~22 are all Dahua 1K cameras
cameras_config = [
    {"ip": "192.168.88.2", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.3", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.4", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.5", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.6", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.7", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.8", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.9", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.10", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.11", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.12", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.13", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.14", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.15", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.16", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.17", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.18", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.19", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.20", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.21", "rtsp_port": "554", "user": "admin", "password": "admin123"},
    {"ip": "192.168.88.22", "rtsp_port": "554", "user": "admin", "password": "admin123"}
]

cameras_ip = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

cameras_selected_area = {
        'ip2': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip3': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip4': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip5': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip6': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip7': [[0,200], [960,0], [1920,200], [1920,1080], [0,1080]],
        'ip8': [[0,150], [960,0], [1920,150], [1920,1080], [0,1080]],
        'ip9': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip10': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip11': [[0,580], [670,0], [1400,0], [1920,640], [1920,1080], [0,1080]],
        'ip12': [[0,570], [950,0], [1920,0], [1920,1080], [0,1080]],
        'ip13': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip14': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip15': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip16': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip17': [[360,560], [800,0], [1200,0], [1800,1080], [360,1080]],
        'ip18': [[0,150], [960,0], [1920,0], [1920,1080], [0,1080]],
        'ip19': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip20': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip21': [[0,0], [1920,0], [1920,1080], [0,1080]],
        'ip22': [[0,0], [1920,0], [1920,1080], [0,1080]]
        }

cameras_map_rect_location = {
        'ip2': [655,613,690,898], 'ip3': [775,613,809,898], 'ip4': [701,613,738,898], 'ip5': [608,613,643,898],
        'ip6': [1349,350,1588,456], 'ip7': [296,916,461,1047], 'ip8': [294,751,460,892],
        'ip9': [711,913,874,1047], 'ip10': [506,915,668,1050], 'ip11': [576,530,908,596], 'ip12': [294,586,461,732],
        'ip13': [919,32,1082,113], 'ip14': [718,32,881,117], 'ip15': [508,29,673,117], 'ip16': [304,33,463,112],
        'ip17': [490,255,903,322], 'ip18': [928,131,1080,266], 'ip19': [1153,219,1381,330],
        'ip20': [938,302,1033,551], 'ip21': [906,606,1059,752], 'ip22': [1049,810,1095,1040]
        }


cv2_font = cv2.FONT_HERSHEY_SIMPLEX

# We pull all cameras at the same time and keep all frames every 180 seconds
global_duration_time = 60 * 10

selected_time_range_one_day = ["0830", "2230"]  # from 8:30am to 22:30pm

# library whole map from top-view with camera markers
global_library_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials/library_map_marker_1920x1080.jpg')

# counting records file that writing the total persons every global_duration_time
global_counting_records = "./counting_records.txt"


###########################################################
'''Functions'''
###########################################################

# calculate two rectangles IOU(intersection-over-union)
def calTwoRectIOU(rectA, rectB):
    [Ax0, Ay0, Ax1, Ay1] = rectA[0:4]
    [Bx0, By0, Bx1, By1] = rectB[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)

# merge two overlapped rectangles into one        
def mergeTwoRect(rectA, rectB):
    [Ax0, Ay0, Ax1, Ay1] = rectA[0:4]
    [Bx0, By0, Bx1, By1] = rectB[0:4]
    return [min(Ax0, Bx0), min(Ay0, By0), max(Ax1, Bx1), max(Ay1, By1)]
    
def merge_overlapping_bbox(bboxs):
    new_bboxs = []
    merge_mark = np.zeros(len(bboxs))
    for i in range(len(bboxs)):
        select_bbox = bboxs[i]
        overlap_flag = False
        for j in range(i+1, len(bboxs)):
            cur_bbox = bboxs[j]
            if calTwoRectIOU(select_bbox, cur_bbox) > 0:
                bbox = mergeTwoRect(select_bbox, cur_bbox)
                merge_mark[i] = 1
                merge_mark[j] = 1
                new_bboxs.append(bbox)
                overlap_flag = True
            else:
                continue
        if not overlap_flag and merge_mark[i]==0:
            new_bboxs.append(select_bbox)
    
    if len(new_bboxs) < len(bboxs):
        return merge_overlapping_bbox(new_bboxs)
    else:
        return new_bboxs
        
        
def remove_outlier_bbox(bboxs, selected_area):
    '''
    Python: checking if point is inside a polygon
    https://stackoverflow.com/questions/16625507/python-checking-if-point-is-inside-a-polygon
    import matplotlib.path as mplPath
    import numpy as np
    poly = [190, 50, 500, 310]
    bbPath = mplPath.Path(np.array([[poly[0], poly[1]],
                         [poly[1], poly[2]],
                         [poly[2], poly[3]],
                         [poly[3], poly[0]]]))
    bbPath.contains_point([200, 100])  # return True
    bbPath.contains_points([200, 100], [-100, 100])  # return array([True, False])
    '''
    if len(bboxs) == 0:
        return bboxs
    else:
        bbPath = mplPath.Path(np.array(selected_area))
        bbox_points = []
        for bbox in bboxs:
            x = int((bbox[0] + bbox[2])/2.0)
            y = int((bbox[1] + bbox[3])/2.0)
            bbox_points.append([x, y])
        is_inside_bool = bbPath.contains_points(np.array(bbox_points))
        left_bboxs = list(np.array(bboxs)[is_inside_bool])
        return left_bboxs


def create_video_cv2(image_list, video_name, scale, fps):
    frames = []
    addFontFlag = True
    # for index, image_name in enumerate(image_list):
    for index, img in enumerate(image_list):
        # if index >= 50: break
        # img = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
        # img = imageio.imread(image_name)  # read RGB-image into RGB-image
        newW, newH = int(img.shape[1] * scale), int(img.shape[0] * scale)
        img = cv2.resize(img, (newW, newH), cv2.INTER_CUBIC)
        
        if addFontFlag:
            img_PIL = Image.fromarray(img)
            font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
            draw = ImageDraw.Draw(img_PIL)
            draw.text((20,20), "E-LearningLab@SJTU", fill = (255,0,0), font=font)  # Chinese is OK
            # img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
            img = np.asarray(img_PIL)
        frames.append(img)
    
    FPS = fps # default is 25
    width, height = frames[0].shape[1], frames[0].shape[0]
    if '.avi' in video_name:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # .avi
    if '.mp4' in video_name:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4  [*'avc1' *'MJPG' *'mp42'] are all wrong 
    out = cv2.VideoWriter(video_name, fourcc, FPS, (width, height))
    for img in frames:
        out.write(img)
    out.release()
    cv2.destroyAllWindows()        


def save_frames(frames, save_path_list):
    scale = 1.0  #0.8
    new_save_path_list = []
    for frame, save_path in zip(frames, save_path_list):
        if frame is not None:  # some camera may be damaged and unreadable
            newW, newH = int(frame.shape[1]*scale), int(frame.shape[0]*scale)
            resized_frame = cv2.resize(frame, (newW, newH), cv2.INTER_CUBIC)
            cv2.imwrite(save_path, resized_frame)
            new_save_path_list.append(save_path)
    return new_save_path_list
    

def is_now_in_selected_time_range():
    time_stamp = datetime.datetime.now().timestamp()             
    cur_time = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d-%H-%M')
    [hour, minute] = cur_time.split('-')[-2:]
    [start_time, end_time] = selected_time_range_one_day
    if int(hour+minute) >= int(start_time) and int(hour+minute) <= int(end_time):
        return True
    else:
        return False


def vis_pose_estimation_in_lib_map(ori_img_paths, json_keypoints, date_stamp, save_res_path):
    map_img = cv2.imread(global_library_map)
    total_num = 0
    image_list = []
    for img_path, json_list in zip(ori_img_paths, json_keypoints):
        ip = int(img_path.split('_')[-1].split('.')[0])
        img_name = 'Dahua1K_IP_' + str(ip) + '.jpg'
        ori_img = cv2.imread(img_path)
                
        selected_area = cameras_selected_area['ip'+str(ip)]
        pts = np.array(selected_area, np.int32).reshape(-1, 1, 2)
        cv2.polylines(ori_img, [pts], True, (0,255,255), 5)
              
        [x0, y0, x1, y1] = cameras_map_rect_location['ip'+str(ip)]
        map_rect = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]])
        temp_map_img = map_img.copy()
        cv2.putText(temp_map_img, date_stamp, (1200, 1000), cv2_font, 2, (0,0,255), 6)
                
        bboxs = []
        for json_dict in json_list:
            bbox = json_dict['bbox']  # this dict has three parts: 'keypoints','bbox','score'
            [x, y, w, h] = list(np.int64(bbox))
            bboxs.append([x, y, x+w, y+h])
        bboxs = merge_overlapping_bbox(bboxs)
        bboxs = remove_outlier_bbox(bboxs, selected_area)
        
        for bbox in bboxs:
            [x_0, y_0, x_1, y_1] = bbox
            cv2.rectangle(ori_img, (x_0, y_0), (x_1, y_1), (0,255,255), 3)
        subtitle_str = 'Camera '+str(ip)+' Person Number: '+str(len(bboxs))
        cv2.putText(ori_img, subtitle_str, (10, 50), cv2_font, 2, (0,0,255), 6)
        cur_save_path = os.path.join(save_res_path, date_stamp)
        if not os.path.exists(cur_save_path):
            os.mkdir(cur_save_path)
        cv2.imwrite(os.path.join(cur_save_path, img_name[:-4]+'_vis.jpg'), ori_img)
        
        total_num += len(bboxs)
        
        for color in [(0,255,255), (0,128,128), (0,255,255), (0,128,128), (0,255,255)]:
            # cv2.fillPoly(temp_map_img, [map_rect], (0, 255, 255))
            cv2.fillConvexPoly(temp_map_img, map_rect, color)
            cv2.putText(temp_map_img, 'IP'+str(ip), (x0, y0+50), cv2_font, 2, (0,255,0), 4)
            cv2.putText(temp_map_img, 'Total Num:'+str(total_num), (1250, 800), cv2_font, 3, color, 6)
            cv2.polylines(ori_img, [pts], True, color, 5)
            img_frame = np.vstack((temp_map_img, ori_img)) 
            image_list.append(img_frame)
    '''Generate the ending of video frames'''
    resH, resW, c = image_list[0].shape
    img_over = np.ones((resH, resW, c), np.uint8)
    img_over = np.uint8(img_over * 128)
    cv2.putText(img_over, 'Made by E-LearningLab@SJTU', (int(resW/4), int(resH*3/7)), cv2_font, 2, (255,255,255), 6)
    cv2.putText(img_over, 'Current Total Person Number '+str(total_num),
        (int(resW/4), int(resH*3.5/7)), cv2_font, 2, (255,255,255), 6)
    cv2.putText(img_over, "For Counting "+date_stamp, (int(resW/4), int(resH*4/7)), cv2_font, 2, (255,255,255), 6)
    image_list += [img_over for i in range(10)]  # fps=5, ending will show 2 seconds
    
    return total_num, image_list
    

def cut_multi_frames_one_time(rtsp_links, start_time, save_frames_path, save_results_path):
    
    # How to keep cut mulit frames in one time? multi-threading
    imgs_name = []
    for [rtsp, id] in rtsp_links:
        img_name = 'Dahua1K_IP_' + str(id) + '.jpg'
        imgs_name.append(img_name)
    
    quit_flag = False
    first_entering_flag = True
    while True and (not quit_flag):
        if 0xff == ord('q'):
            quit_flag = True
            break
            
        if not is_now_in_selected_time_range():
            continue  # dead loop; threading sleeping
            
        # We get out one frame every 5 minutes
        if (time.time() - start_time) > global_duration_time or first_entering_flag:
            first_entering_flag = False
            start_time = time.time()
            
            all_frames = []
            for [rtsp, id] in rtsp_links:
                cap = cv2.VideoCapture(rtsp)
                ret, frame = cap.read()
                cap.release()
                all_frames.append(frame)
                
            time_stamp = datetime.datetime.now().timestamp()             
            cur_time = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d-%H-%M')
            cur_save_path = os.path.join(save_frames_path, cur_time)
            os.mkdir(cur_save_path)
            save_path_list = [os.path.join(cur_save_path, img_name) for img_name in imgs_name]
             
            '''Step 1: save original frames cut from cameras'''
            print("Step 1 --->>> Save all cameras' frame in one same time:", cur_time)
            new_save_path_list = save_frames(all_frames, save_path_list)
            '''Step 2: do pose estimation in those frames'''
            print("Step 2 --->>> Do pose estimation in those selected frames...")
            json_keypoints = pose_pred.inference(imgs_path=new_save_path_list, model_path=os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "pifpaf_slim_v3/models/resnet101_block5-pif-paf.pkl"), use_gpu=False)
            '''Step 3: counting persons and save vis_results'''
            print("Step 3 --->>> Deal with pose estimation results and get person counting...")
            total_num, image_list = vis_pose_estimation_in_lib_map(new_save_path_list, 
                json_keypoints, cur_time, save_results_path)
            '''Step 4: generate a gif or video with image_list'''
            print("Step 4 --->>> Generate short video to show counting result...")
            video_name = os.path.join(save_results_path, cur_time, 'Dahua1K_Counting_'+str(total_num)+'.mp4')
            create_video_cv2(image_list, video_name, scale=0.8, fps=5)
            
            
            print(cur_time, time.time(), " **** Synchronous cameras successfully! **** ")
            print("--->>>", cur_time, 'has been finished.', 'Total Persons Are', total_num)
            
            record_file = open(global_counting_records, 'a')  # append write and not be able to read
            record_file.write("Time Stamp: " + cur_time + ", Room Location: B300, Total Persons: " + str(total_num) + "\n")
            record_file.close()
 
            
def rtsp_get_full_frames_xintu_v1():
    # Template for Dahua "rtsp://user:password@ip:port/cam/realmonitor?channel=1&subtype=0"
    rtsps = []
    for i, camera in enumerate(cameras_config):
        main_rtsp = "rtsp://" + camera["user"] + ":" + camera["password"] + "@" + camera[
            "ip"] + ":" + camera["rtsp_port"] + rtsp_postfix["Dahua"][0]
        ip_id = camera["ip"].split('.')[-1]
        rtsps.append([main_rtsp, ip_id])

    save_frames_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames/")
    if not os.path.exists(save_frames_path):
        os.mkdir(save_frames_path)
        
    save_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis_results/")
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    
    cut_multi_frames_one_time(rtsps, time.time(), save_frames_path, save_results_path)
    
    
if __name__ == '__main__':
    rtsp_get_full_frames_xintu_v1()
