import pandas as pd
import os
from dotenv import load_dotenv, dotenv_values
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model_List import model_paths
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_prompt(text):
    prompt = f"""
    “Combine all eight image descriptions of the same 3D model into a single compound sentence by extracting specific details, and removing long, repetitive, and uncertain phrases. An example of the desired approach is provided below:
    Descriptions:
    The focussed object is a 3D model of an airplane. It has a white body with a blue and white tail. The wings are black and have a blue stripe. The nose of the airplane is white and has a blue and white stripe. The airplane is flying in a straight line, with the nose pointing towards the sky. The background is a dark gray color, providing a contrast to the airplane. There are no texts or other objects in the image. The focus is on the airplane
    The focussed object in the image is a 3D model of an airplane. It has a white body with a blue and white tail. The wings are gray and have a white stripe. The nose of the airplane is black and has a blue and white logo. The airplane is flying in a straight line, with the nose pointing towards the sky. The background is a dark gray color, providing a contrast to the airplane. There are no texts or other objects in the image. The focus is"
    The focussed object in the image is a white airplane with a blue tail. The airplane is flying through the air, with its wings spread out and its tail pointing downwards. The background of the image is a gray color, providing a neutral backdrop that allows the focus of the object to stand out. There are no other objects or text in the image, and the relative position of the airplane to the background is such that it is the main subject of the image."
    The focussed object in the image is a white airplane with a blue tail. The airplane is flying through the air, with its wings spread out and its tail pointing downwards. The background of the image is a gray color, providing a neutral backdrop that allows the focus of the object to stand out. There are no other objects or text in the image, and the relative position of the airplane to the background is such that it is the main subject of the image."
    The focussed object in the image is a white airplane with a blue tail. The airplane is moving forward, with its wings spread out in a V shape. The background is a dark gray color, providing a contrast to the airplane. There are no other objects or text in the image. The focus is on the airplane, with no other objects or text in the image. The airplane is the only object in the image, and it is the main subject of the image. The background is dark gray"
    The focussed object in the image is a white airplane with a blue and white tail. The airplane is flying through a gray sky. The airplane has four engines and is equipped with a landing gear system. The landing gear is attached to the airplane's nose. The airplane is moving forward, with its nose pointing towards the sky. The airplane's tail is visible, and it is connected to the landing gear system. The airplane's body is mostly white, with some blue and white accents. The airplane"
    The focussed object in the image is a white airplane with a blue and white body. The airplane has four wings and four engines. The wings are spread out, and the engines are located at the top of the wings. The body of the airplane is white, and it has a blue and white color scheme. The airplane is flying in the sky, and it appears to be moving forward. The background of the image is dark, and there are no other objects or text present. The focus of"
    The focussed object in the image is a white airplane with a blue tail. It has four engines and four wheels. The airplane is flying in the sky and appears to be in motion. The background is dark and does not provide any additional context or information about the airplane.
    Caption: a white airplane with a blue tail, striped wings, four engines and four wheels
    Descriptions: {text}
    Caption:”
    """
    return prompt

ShapeNetCoreDescriptions = {
    'Class': [],
    'Subclass': [],
    'Caption': [],
}
output_file_path = './text/fusion.csv'
df = pd.read_csv('./text/captions.csv')

for path in model_paths[3:4]:
    c,s = path.split('/')[2:4]
    descriptions = df[(df['Class']==int(c[1:])) & (df['Subclass']==s)]['Caption'].to_list()
    ShapeNetCoreDescriptions['Class'].append(str(c))
    ShapeNetCoreDescriptions['Subclass'].append(str(s))
    response = client.chat.completions.with_raw_response.create(
        messages=[{
            "role": "user", "content": get_prompt('\n'.join(descriptions)),
        }],
        model="gpt-4o-mini",
        temperature=0.1,
    )
    completion = response.parse()
    caption = str(completion.choices[0].message.content)
    ShapeNetCoreDescriptions['Caption'].append(caption)

caption_df = pd.DataFrame(ShapeNetCoreDescriptions)
caption_df.to_csv(output_file_path, index=False)