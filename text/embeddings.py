import pandas as pd
from openai import OpenAI
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ast import literal_eval
import plotly.graph_objects as go

df = pd.read_csv('./captions.csv')

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embedding_model = "text-embedding-3-small"
def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
embedding_dir = './embedding.csv'
df['Embedding'] = df['Caption'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv(embedding_dir, index=False)

# Visualising the text embeddings on a 3D plot
datafile_path = "embedding_dir"
df = pd.read_csv(datafile_path)
subclasses = df['Subclass']
matrix = np.array(df['Embedding'].apply(literal_eval).to_list())
# a = get_embedding("hi", model=embedding_model)
# print(a)
tsne = TSNE(n_components=3, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)
# Mapping subclass identifiers to numeric values for coloring
unique_subclasses = np.unique(subclasses)
subclass_map = {label: idx for idx, label in enumerate(unique_subclasses)}
color_labels = subclasses.map(subclass_map)  # Convert labels to integers
# Normalize your labels to be between 0 and 1 for a continuous color scale
color_labels_normalized = (color_labels - color_labels.min()) / (color_labels.max() - color_labels.min())
# Create the 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=vis_dims[:, 0],
    y=vis_dims[:, 1],
    z=vis_dims[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=color_labels_normalized,  # Use normalized labels for color scale
        colorscale='Viridis',  # You can choose any available colorscale
        # colorbar=dict(
        #     title='Subclass',
        #     titleside='top',
        #     tickmode='array',
        #     tickvals=np.linspace(0, 1, len(unique_subclasses)),
        #     ticktext=[str(label) for label in unique_subclasses]
        # ),
        opacity=0.7
    )
)])
fig.update_layout(title='3D TSNE Visualization of Text Embeddings', scene=dict(xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Dimension 3'))
fig.write_html('3D_plot_text_embeddings.html')
# fig.show()