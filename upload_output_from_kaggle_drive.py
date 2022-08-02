# I was facing the same problem and was able download files from Kaggle to Colab then move to Google Drive. 
# For example, if the current directory is /kaggle/working and the file to move is processed_file.zip then,

# From Kaggle
from IPython.display import FileLink

FileLink(r'processed_file.zip') #This will generate a link, https://....kaggle.net/...../processed_file.zip

# From Colab downlaod files to colab
!wget "https://....kaggle.net/...../processed_file.zip"

#Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy File to Google Drive
!cp "/content/processed_file.zip" "/content/drive/My Drive/workspace"
