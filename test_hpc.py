from torch.hub import download_url_to_file
from torchaudio.datasets.utils import extract_archive

url = "https://drive.google.com/file/d/1BFtutlPE8mMY3kawb3zN9hXalmGqxyoE/view?usp=share_link"
path = "/zhome/56/e/155505/Desktop/"

download_url_to_file(url, path)

extract_archive(path + "dataset_zip.zip",  path + "data")


