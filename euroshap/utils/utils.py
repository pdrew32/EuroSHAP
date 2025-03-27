import requests


def report_filesize(url):
    response = requests.head(url)
    filesize = int(response.headers.get("Content-Length", 0))
    print("File size in MB:", filesize / (1024*1024))
    return