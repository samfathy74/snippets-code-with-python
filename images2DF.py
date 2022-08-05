from PIL import Image
import numpy as np
import cv2

def img2df(image_list, resize_shape=None, output_name='CSV_File_Output.csv'):
  for img in image_list:
    # Read image and transfer to gray
    gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # transform image array to pixels array
    img_pix = Image.fromarray(gray_img)
    # resize image
    if resize_shape is not None:
      img_pix = np.array(img_pix.resize(resize_shape, Image.ANTIALIAS))
    # apply flatten above image
    flat_img_array = (img_pix.flatten())
    # do reshape to image
    img_array  = flat_img_array.reshape(1, -1)

    # save image in csv file
    with open(output_name, 'ab') as f:
      np.savetxt(f, img_array, delimiter=",")
    print(f"Transforamtion {img} Done...")
