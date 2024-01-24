import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import shutil 
def download_images(query, output_folder, limit=20):
    search_url = f'https://www.google.com/search?q={query}&tbm=isch'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    image_urls = []
    for img_tag in soup.find_all('img', {'class': 'rg_i'}):
        if 'data-src' in img_tag.attrs:
            image_url = img_tag['data-src']
            image_urls.append(image_url)

        if len(image_urls) >= limit:
            break

  
    for i, image_url in enumerate(image_urls):
        response = requests.get(image_url)
        image_path = os.path.join(output_folder, f'image_{i+1}.jpg')
        with open(image_path, 'wb') as img_file:
            img_file.write(response.content)

def process_image(image_path, category, images_folder):
    destination_folder = os.path.join(images_folder, category)
    shutil.move(image_path, destination_folder)

def interactive_review(images_folder):
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)


        img = Image.open(image_path)
        img.show()


        user_response = input(f"Is this image correct, incorrect, or delete? (y/n/delete): ")

        if user_response.lower() == 'y':

            process_image(image_path, 'correct', images_folder)
        elif user_response.lower() == 'n':
          
            process_image(image_path, 'incorrect', images_folder)
        elif user_response.lower() == 'delete':
         
            os.remove(image_path)

if __name__ == "__main__":
    search_query = "wrong handstand"
    download_folder = "handstand_images"

 
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_images(search_query, download_folder)


    for folder_name in ['correct', 'incorrect']:
        folder_path = os.path.join(download_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


    interactive_review(download_folder)
