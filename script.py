! pip install -q kaggle
import os
from google.colab import files

def import_data_from_kaggle(url_dataset, OUTPUT_FOLDER="./dataset/", unzip=True):
  """
  url_dataset String : dataset url from kaggle for example ('https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data')
  OUTPUT_FOLDER String : destination directory for downloaded files ("./dataset/")
  unzip Boolean : if true extract files from zip file 
  """
  # upload APi-key json file download it from your profile from kaggle
  if not os.path.isfile('kaggle.json'):
    print("[+] Upload 'Kaggle.json' api for credentials")
    print(f"[*] You can download it from kaggle profile account 'https://www.kaggle.com/{input('username of kaggle')}/account'\n\n")
    files.upload()
  else:
    print('file already exist')

  os.system("mkdir ~/.kaggle")
  os.system("cp kaggle.json ~/.kaggle/")

  url_list = url_dataset.split('/')

  if unzip:
    os.system("kaggle datasets download "+url_list[-2]+"/"+url_list[-1]+" -p "+OUTPUT_FOLDER+url_list[-1]+" --unzip")
  else:
    os.system("kaggle datasets download "+url_list[-2]+"/"+url_list[-1]+" -p "+OUTPUT_FOLDER+url_list[-1])
