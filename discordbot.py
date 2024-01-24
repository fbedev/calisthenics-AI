import discord
from discord.ext import commands
import cv2
import torch
from torchvision import transforms
from hs_definition import CalisthenicsNet 
import os
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

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

@bot.command()
async def calisthenics(ctx):
  
    if not ctx.message.attachments:
        await ctx.send("Please attach at least one image file.")
        return

   
    image_attachment = ctx.message.attachments[0]

    image_path = f"temp_image_{image_attachment.filename}"
    await image_attachment.save(image_path)

    model_path = "calisthenics_hs.pth"
    model = load_model(model_path)

   
    input_image = preprocess_image(image_path)
    prediction = predict(model, input_image)

  
    result_message = f'Prediction for {image_attachment.filename}: {prediction:.2%}'
    await ctx.send(result_message)


    os.remove(image_path)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')


bot.run('token')
