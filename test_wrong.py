
import os
from tqdm import tqdm
from predict import AgeEstimator 
import matplotlib.image as mpimg
from PIL import Image  
import os  
import glob  


import os  
  

def get_frame(path, model):
    count = 1
    filename = 'wrong'
    model.count_zero(filename)
      
    jpg_files = []  
    png_files = []  
    
    for root, dirs, files in os.walk(path):  
        for file in files:  
            if file.endswith(".jpg"):  
                jpg_files.append(os.path.join(root, file))  
            elif file.endswith(".png"):  
                png_files.append(os.path.join(root, file))

    all_photo_files = jpg_files + png_files  
      
    for photo_file in tqdm(all_photo_files):  
        image = Image.open(photo_file).convert('RGB')
        predicted_image, underage = model.predict(image)
        # mpimg.imsave(f'wrong/output_{filename}_{count}.png',predicted_image)
        if underage:
            mpimg.imsave(f'test_0327/underage/underage_{count}.png',predicted_image)
        else:
            mpimg.imsave(f'test_0327/adult/adult_{count}.png',predicted_image)
        count += 1
    # records = model.get_record()
    # with open('0326_test_jsy_output.txt', 'a') as f:
    #     f.write(records)  



def main(path):
    model = AgeEstimator()
    get_frame(path, model)

if __name__=="__main__":
    path = '/DATA/jupyter/personal/age_classifier/results/wrong_0325'
    main(path)