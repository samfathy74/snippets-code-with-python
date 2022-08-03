import gdown
def download_from_drive(id, output_path='./', folder=False, folder_limited=False):
  """
  id String: is a unique id for every file and folder is a part of drive link is example(https://drive.google.com/uc?id={{'1FxvmwTrYZsMyCMfH_mcrLJ2oStXABQG_'}})
  output_path='./output/' String: is output path of downloaded files,
  folder=False :Boolean if you need to download folder from drive ,
  """
  EXTENTION='zip'

  if not folder:
    if not os.path.exists(output_path):
      os.mkdir(output_path)

    downloaded_file = gdown.download(id=id, output=output_path)
    # print('\n|>>>>',downloaded_file)

    if downloaded_file.split('.')[-1] == EXTENTION:
      print("extract files...")
      gdown.extractall(path=downloaded_file, to=output_path)
  else:
    if folder_limited:
      '''if u have a problem with folder_limited you should run this command external'''
      out = os.system(f"gdown --remaining-ok -q --folder -O {output_path} --id {id} ")
      if out>0:
        print("download completed")
    else:
      gdown.download_folder(id=id, output=output_path, remaining_ok=True)  
