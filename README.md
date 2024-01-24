calisthenics AI
## Requirements
- Python 3.7 or higher
- PyTorch
- discord.py
- OpenCV
- torchvision
- requests
- BeautifulSoup
- Pillow

Install the required packages using:
```bash
#install requirements
pip install -r requirements.txt
#get AI pth file, you can also get it from the calisthenics_hs.pth
python3 hs_AI.py

#more
python3 hs_google.py
to get 20 handstand picture to the folder(for training)

# Usage
python3 hs_inference.py --model model_weights.pth --image (path to image)
