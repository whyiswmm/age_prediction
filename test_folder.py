
import os
from tqdm import tqdm
from predict import AgeEstimator 
import matplotlib.image as mpimg
from PIL import Image  
import os  
import glob  


def get_frame(path, model):
    count = 1
    filename = 'test_jsy_0326'
    model.count_zero(filename)

    jpg_files = glob.glob(os.path.join(path, "*.jpg"))  
    png_files = glob.glob(os.path.join(path, "*.png"))  
      
    all_photo_files = jpg_files + png_files  
      
    for photo_file in tqdm(all_photo_files):  
        image = Image.open(photo_file)
        predicted_image, _ = model.predict(image)
        # mpimg.imsave(f'test_jsy_0327/output_adults_{count}.png',predicted_image)
        count += 1
    records = model.get_record()
    with open('0327_test_jsy_output.txt', 'a') as f:
        f.write(records)  



def main(path):
    model = AgeEstimator()
    get_frame(path, model)

if __name__=="__main__":
    path = '/DATA/jupyter/personal/dataset/test_from_jsy/val_data_holepicture/adults'
    main(path)
    path = '/DATA/jupyter/personal/dataset/test_from_jsy/val_data_holepicture/minors'
    main(path)