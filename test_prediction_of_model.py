def check_model(test_data:np, test_labels:np, model):
  images = []

  # randomly select a few testing characters
  for i in np.random.choice(np.arange(0, len(x_test)), size=(100,)):
    # predicte the character
    probs = model.predict(x_test[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    # the image
    image = (x_test[i] * 255).astype("uint8")
    color = (0, 255, 0)

    # otherwise, prediction is incorrect
    if prediction[0] != np.argmax(test_labels[i]):
      color = (255, 0, 0)

    # merge the channels into one image, resize the image from 32x32
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

  #       if you work with Arabic Language use(remove comments):
                    #   text = arabic_reshaper.reshape(label)
                    #   font = ImageFont.truetype(fontpath, 30)
                    #   img_pil = Image.fromarray(image)
                    #   draw = ImageDraw.Draw(img_pil)
                    #   draw.text(xy=(5, 15), text=text[::-1], font = font, fill=color)
                    #   image = np.array(img_pil)

  #      otherwise: if you work with any other language use(remove comments):
                   # cv2.putText(image, reshaped_text[::-1], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # add the image to our list of output images
    images.append(image)


  # construct the montage for the images
  montage = build_montages(images, (100, 100), (7, 7))[0]
  # show the output montage

  plt.figure(figsize=(15,7))
  plt.imshow( montage)
  plt.show()
