def Generate_Images(dir_of_original_images, num_copies=100, output_dir='./output_copies/', save_format='jpg', target_size=(100, 100), color_mode='rgb'):
  datagen = ImageDataGenerator(
                        brightness_range=(0.6, 0.4), 
                        channel_shift_range=0.5, 
                        rotation_range=10,
                        zoom_range=0.1,
                        width_shift_range=0.2,
                        height_shift_range=0.1,
                        shear_range=0.20,
                        vertical_flip=True,
                        horizontal_flip=True)
  
  train_generator = datagen.flow_from_directory(
                  directory= dir_of_original_images, #input directory
                  target_size=target_size, # resize to this size
                  color_mode=color_mode, # for coloured images
                  batch_size=1, # number of images to extract from folder for every batch
                  class_mode="binary", # classes to predict
                  seed=2020, # to make the result reproducible
                  save_to_dir=output_dir, #output folder
                  save_prefix=f'aug17406_', #output name 
                  save_format = save_format # output extentaion
                  )
  
  for i in range(num_copies):
    next(train_generator)[0].astype('uint8')
