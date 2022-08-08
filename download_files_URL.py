from requests import get  # to make GET request
def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


if __name__ == "__main__":
  download("https://i0.wp.com/www.annuair.ma/wp-content/uploads/2021/03/5-important-ways-learn-alphabets-teach-your-kids-correct-pronunciation.jpg?fit=800%2C350&ssl=1","image.jpg")
