import argparse
import cv2
import torch
from torchvision import transforms
from hs_definition import CalisthenicsNet  

def load_model(model_path):
    model = CalisthenicsNet()
    
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preprocessed_image = transform(image)
    return preprocessed_image

def predict(model, image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        prediction = torch.sigmoid(output).item()
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Calisthenics AI Inference')
    parser.add_argument('--model', required=True, help='Path to the trained model weights')
    parser.add_argument('--image', required=True, help='Path to the input image for inference')
    args = parser.parse_args()

    model = load_model(args.model)
    input_image = preprocess_image(args.image)
    prediction = predict(model, input_image)

    print(f'Prediction for {args.image}: {prediction:.2%}')

if __name__ == "__main__":
    main()
