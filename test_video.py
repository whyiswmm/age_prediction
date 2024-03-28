import urllib.request
import requests
import cv2
import os
import argparse
import os
if not os.path.exists(os.path.join("runs", "predict")):
    os.makedirs(os.path.join("runs", "predict"))
from predict import AgeEstimator 
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image  
from tqdm import tqdm

#upload:   aws --endpoint-url=http://jssz-inner-boss.bilibili.co s3 cp 文件路径 s3://bucket_name/目标路径 --recursive
#download:   aws --endpoint-url=http://jssz-inner-boss.bilibili.co s3 cp s3://bucket_name/目标路径 文件路径  --recursive
# http://jssz-inner-boss.bilibili.co/cv_data_storage/age_estimation/

# aws --endpoint-url=http://jssz-inner-boss.bilibili.co s3 cp ./test.txt s3://cv_data_storage/age_estimation/ --recursive


def get_frame(filename, model):
    directory = "./video/"
    video_name = os.path.join(directory, filename+'.mp4')
    if not os.path.exists(video_name):
        return
    frame_directory = os.path.join('0326test', filename)
    if os.path.exists(frame_directory):
        return
    os.makedirs(frame_directory)    
    video = cv2.VideoCapture(video_name)
    frame_count = 1
    model.count_zero(filename)

    while frame_count <= 20:
        # 读取视频的下一帧
        ret, frame = video.read()
        # 如果读取成功
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            # 将ndarray转换为PIL Image对象  
            image = Image.fromarray(frame_rgb) 
            predicted_image, record_or_not = model.predict(image)
            if record_or_not:
                mpimg.imsave(f'{frame_directory}/output_{filename}_frame_{frame_count}.png',predicted_image)
            frame_count += 1
        else:  
            break
    video.release()
    records = model.get_record()
    with open('0326_output.txt', 'a') as f:
        f.write(records)  

#下载一秒一帧视频
def get_1s1f_video(filename):
    directory = "./video/"
    local_path = os.path.join(directory, filename+'.mp4')
    if os.path.exists(local_path):
        return
    base_url_1s1f = "http://bvcflow-executor3.bilibili.co?r=info&method=get_info&key=videoframes&flowid="
    url = requests.get(base_url_1s1f + filename).text
    url = eval(url.replace("\/", "/"))[0]
    if not url:
        # print(f"can't    get  {filename}")
        return
    # print('url', url)
    download_file(url, local_path)
    
    
def download_file(url, local_path):
    """
    download video to local_path
    """
    with requests.get(url, stream=True) as r:
        contentlen = int(r.headers["content-length"])
        downSize = 0
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                downSize += len(chunk)
                if downSize >= contentlen:
                    break
    if downSize > 0:
        return local_path


def main(file):
    model = AgeEstimator()
    with open(file, "r") as f:
        lines = f.readlines()
        last_column = [line.split()[-1] for line in lines]
        for i in tqdm(last_column):
            get_1s1f_video(i)
            get_frame(i, model)
            # print(f'finished read {i}')

if __name__=="__main__":
    path = '2000sample.txt'
    main(path)