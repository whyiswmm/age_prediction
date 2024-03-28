
import os
from tqdm import tqdm
from predict import AgeEstimator 
import matplotlib.image as mpimg
from PIL import Image  
import os  
import glob  
from transformers import AutoImageProcessor


def get_frame(path, model):
    # model_checkpoint = '/DATA/jupyter/personal/age_classifier/nateraw/vit-age-classifier'
    # image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    image = Image.open(path).convert("RGB")
    # encoding = image_processor(image.convert("RGB"), return_tensors="pt")
    predicted_image, _ = model.predict(image)
    # if underage:
    mpimg.imsave(f'output.png',predicted_image)
    # else:
    #     mpimg.imsave(f'test_0327/adult/output.png',predicted_image)

    # records = model.get_record()
    # with open('0326_test_jsy_output.txt', 'a') as f:
    #     f.write(records)  



def main(path):
    model = AgeEstimator()
    get_frame(path, model)

if __name__=="__main__":
    path = 'WechatIMG16855.jpg'
    main(path)