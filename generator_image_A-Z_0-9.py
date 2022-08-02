# important things 
# import fonts
os.system("git clone https://github.com/gasharper/linux-fonts.git")
# delete anomalies files from dir
os.system("rm -r ./linux-fonts/README.md")
os.system("rm -r ./linux-fonts/.git")
os.system("rm -r ./linux-fonts/install.sh")
# some fonts not good
os.system("rm -r ./linux-fonts/WEBDINGS.ttf ./linux-fonts/symbol.ttf ./linux-fonts/wingding.ttf ./linux-fonts/WINGDNG2.ttf ./linux-fonts/WINGDNG3.ttf ./linux-fonts/mtextra.ttf")
# create input dir and 
os.system("mkdir ./CaptchaDataset/")

# to create Image for every font
for font in glob.glob("fonts\\*"): 
    fontTypes.append(os.path.abspath(font))

for index, character in enumerate(charactersList):
    path = os.path.join(parentPath, character)

    if not os.path.exists(path):
        os.mkdir(path)
     
    for imageCounter in range(len(fontTypes)):
        for repeats in range(100): # Number of Images
#             img = Image.new('1', (32, 32), color = 'black')
            fnt = ImageFont.truetype('FONTS_DIR' + fontTypes[imageCounter], random.randint(20, 50))
            w, h = fnt.getsize(character)
            img_w, img_h = w + 10, h + 10  # Add 20 pixels padding (assume 10 pixels from each side).
            img = Image.new('L', (img_w, img_h), color='black')  # Replace '1' with 'L' (8-bit pixels, black and white - we fill 255 so we can't use 1 bit per pixel)
            d = ImageDraw.Draw(img)
            d.text(((img_w-w)/2, (img_h-h)/2), character, font=fnt, fill=255, align="center") # TO ALIGN CHARACTER IN CENTER
            img.save(f"{path}/{imageCounter}{repeats}_{character}.jpg")
