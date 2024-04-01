
import os
from tqdm import tqdm
from predict import AgeEstimator 
import matplotlib.image as mpimg
from PIL import Image  
import os  
import glob  


def get_frame(path, model):
    count = 1
    filename = 'yisi'
    model.count_zero(filename)

    jpg_files = glob.glob(os.path.join(path, "*.jpg"))  
    png_files = glob.glob(os.path.join(path, "*.png"))  
      
    all_photo_files = jpg_files + png_files  
      
    for photo_file in tqdm(all_photo_files):  
        image = Image.open(photo_file).convert("RGB")
        predicted_image, save = model.predict(image)
        if save:
            mpimg.imsave(f'yangli/{count}.png',predicted_image)
            # os.remove(photo_file)
        count += 1
    records = model.get_record()
    with open('yisi_0329.txt', 'a') as f:
        f.write(records)  



def main(path):
    model = AgeEstimator()
    get_frame(path, model)

if __name__=="__main__":
    path = 'yisi_0329'
    main(path)
    # path = '/DATA/jupyter/personal/dataset/test_from_jsy/val_data_holepicture/minors'
    # main(path)