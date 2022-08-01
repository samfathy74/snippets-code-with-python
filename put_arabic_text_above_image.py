def put_ar_text(img, text, cor=(0,0), font_size=32, fontpath="./linux-fonts/arial.ttf", fill_color=(255,255,0), windows=False):
  os.system("git clone https://github.com/gasharper/linux-fonts.git")
  os.system("pip install arabic_reshaper")
  os.system("pip install python-bidi")
  font = ImageFont.truetype(fontpath, font_size)
  img_pil = Image.fromarray(img)
  draw = ImageDraw.Draw(img_pil)
  reshaped_text = arabic_reshaper.reshape(text)
  bidi_text = get_display(reshaped_text) 
  draw = ImageDraw.Draw(img_pil)
  draw.text(cor, bidi_text, font = font, fill=fill_color)
  img = np.array(img_pil)
  if windows:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return img
