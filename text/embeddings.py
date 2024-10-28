import pandas as pd
from openai import OpenAI
import os
import numpy as np
from sklearn.manifold import TSNE
from ast import literal_eval
import plotly.graph_objects as go
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from dotenv import load_dotenv

# Plotting 3D TSNE
def viz_tsne(embedding_dir):
    df = pd.read_csv(embedding_dir)
    subclasses = df['Subclass']
    matrix = np.array(df['Embedding'].apply(literal_eval).to_list())
    tsne = TSNE(n_components=3, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    unique_subclasses = np.unique(subclasses)
    subclass_map = {label: idx for idx, label in enumerate(unique_subclasses)}
    color_labels = subclasses.map(subclass_map)
    color_labels_normalized = (color_labels - color_labels.min()) / (color_labels.max() - color_labels.min())
    fig = go.Figure(data=[go.Scatter3d(
        x=vis_dims[:, 0],
        y=vis_dims[:, 1],
        z=vis_dims[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_labels_normalized,
            colorscale='Viridis',
            opacity=0.7
        )
    )])
    fig.update_layout(title='3D TSNE Visualization of Text Embeddings', scene=dict(xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Dimension 3'))
    fig.write_html('3D_plot_text_embeddings.html')
    fig.show()

def get_embedding(text, client):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=config.text_embedding_model).data[0].embedding

def main():
    df = pd.read_csv('./text/captions_50.csv')
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API'))
    df['Embedding'] = df['Caption'].apply(lambda x: get_embedding(x, client))
    df.to_csv(config.embeddings_dir, index=False)
    # viz_tsne(embedding_dir) # Visualizing the TSNE

if __name__ == '__main__':
    main()