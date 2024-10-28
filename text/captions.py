import pandas as pd
import os
from dotenv import load_dotenv, dotenv_values
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from Model_List import model_paths
from openai import OpenAI
from ShapeNetCore import get_random_models

def get_prompt(text):
    prompt = f"""
    Compress the given description into only one sentence. Remove unrelated phrases and discard repeated information. Example:
    Text: The focussed object in the image is a white, circular object with a curved top. It has a smooth surface and a slight curve at the top. The object appears to be made of a material that gives it a slightly textured appearance. The color of the object is white, and it has a reflective quality to it. The object is positioned centrally in the image, and it is the only object in the frame. There are no other objects or texts visible in the image.
    Summary: a white, circular object with a smooth and reflective curved top.

    Text: {text}
    Summary:
    """
    return prompt

def main():
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API"))
    ShapeNetCoreDescriptions = {
        'Class': [],
        'Subclass': [],
        'Caption': [],
    }
    df = pd.read_csv(config.descriptions_dir)
    # for path in model_paths:
    for path in get_random_models()[:]:
        sub_dirs = path.split('/')
        c, s = None, None
        if len(sub_dirs) == 4:
            c,s = path.split('/')[2:4]
        else:
            c,s = path.split('/')
        description = df[(df['Class']==int(c[1:])) & (df['Subclass']==s)]['Description'].iloc[0]
        ShapeNetCoreDescriptions['Class'].append(str(c))
        ShapeNetCoreDescriptions['Subclass'].append(str(s))
        response = client.chat.completions.with_raw_response.create(
            messages=[{
                "role": "user", "content": get_prompt(str(description)),
            }],
            model=config.text_captioning_model,
            # model="text-embedding-3-small",
            temperature=0,
        )
        # print(response.headers.get('x-request-id'))
        completion = response.parse()
        caption = str(completion.choices[0].message.content)
        ShapeNetCoreDescriptions['Caption'].append(caption)

    caption_df = pd.DataFrame(ShapeNetCoreDescriptions)
    caption_df.to_csv(config.captions_dir, index=False)

if __name__ == '__main__':
    main()