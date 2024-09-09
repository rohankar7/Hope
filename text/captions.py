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
    # Prepare the prompt with few-shot examples
    prompt = f"""
    Compress the given description into only one sentence. Remove unrelated phrases and discard repeated information. Example:
    Text: The focussed object in the image is a white, circular object with a curved top. It has a smooth surface and a slight curve at the top. The object appears to be made of a material that gives it a slightly textured appearance. The color of the object is white, and it has a reflective quality to it. The object is positioned centrally in the image, and it is the only object in the frame. There are no other objects or texts visible in the image.
    Summary: a white, circular object with a smooth and reflective curved top.

    Text: {text}
    Summary:
    """
    return prompt

ShapeNetCoreDescriptions = {
    'Class': [],
    'Subclass': [],
    'Caption': [],
}
output_file_path = './text/all_captions.csv'
df = pd.read_csv('./text/descriptions.csv')

for path in model_paths:
    c,s = path.split('/')[2:4]
    description = df[(df['Class']==int(c[1:])) & (df['Subclass']==s)]['Description'].iloc[0]
    ShapeNetCoreDescriptions['Class'].append(str(c))
    ShapeNetCoreDescriptions['Subclass'].append(str(s))
    response = client.chat.completions.with_raw_response.create(
        messages=[{
            "role": "user", "content": get_prompt(str(description)),
        }],
        model="gpt-4o-mini",
        # model="text-embedding-3-small",
        temperature=0,
    )
    # print(response.headers.get('x-request-id'))
    completion = response.parse()
    caption = str(completion.choices[0].message.content)
    ShapeNetCoreDescriptions['Caption'].append(caption)

caption_df = pd.DataFrame(ShapeNetCoreDescriptions)
caption_df.to_csv(output_file_path, index=False)