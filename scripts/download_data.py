import os
import gdown


class DataDownloader:
    def __init__(self, url: str, target_folder: str = './data'):
        self.url = url
        self.target_folder = target_folder
        self.download_data()

    def download_data(self):
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

        # Get the file ID from the Google Drive link
        output = os.path.join(self.target_folder, 'data-final.csv')  # Ensure correct path
        gdown.download(self.url, output, quiet=False)
        print(f"Dataset downloaded to {output}")
