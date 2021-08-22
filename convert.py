from PIL import Image
from tqdm import tqdm
import os

TEST_DIR = 'test'
save_p = 'test1/'
i='0'
for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, img)

    im1 = Image.open(path)
    print(path)
    im1.save(save_p+i+".jpg")
    temp=(int(i)+1)
    i=str(temp)

