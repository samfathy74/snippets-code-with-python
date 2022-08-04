
> `image_array.reshape((image_array.shape[0], 32, 32)).astype('uint8')`   # reshape to image

> `cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)`      # Rotating the images.

> `np.flip(image_array, 1)`                               # Flipping the images

> `image_array.astype('float32')/255`                     # Here we normalize our images.

> `cv2.merge([image] * n)`                        # merge the channels into one image, resize the image to (n dim)
