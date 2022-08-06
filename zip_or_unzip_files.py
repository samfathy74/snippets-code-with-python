import shutil
import tempfile

def archive_files(file_path, target_path='./', format='zip',  unzip=False):
  if unzip:
    shutil.unpack_archive(filename=file_path, extract_dir=target_path, format=format)
    print(f'Completed unpack archive in {shutil.os.getcwd()}')
  else:
    tmp_dir = tempfile.mkdtemp()
    if not os.path.isdir(file_path):
      shutil.copy2(file_path, tmp_dir)
      shutil.make_archive(root_dir=tmp_dir, base_name=file_path, format=format, verbose=1)
      shutil.rmtree(tmp_dir)
    else:
      shutil.make_archive(root_dir=file_path, base_name=file_path, format=format, verbose=1)
      
    print(f'Completed make archive in {shutil.os.getcwd()}')
    
