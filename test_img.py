
from predict import AgeEstimator 
import matplotlib.image as mpimg
from PIL import Image  

if __name__=="__main__":
    path = 'WechatIMG16855.jpg'
    model = AgeEstimator()
    image = Image.open(path).convert("RGB")
    predicted_image, _ = model.predict(image)
    mpimg.imsave(f'output.png',predicted_image)