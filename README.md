# Text to 3D Model Generation using Diffusion Techniques

## Methodology

### Preparing the Dataset
1) Run ./blender/blender_captions.py: This will render 8 holistic images of every 3D model and save it locally
2) Run ./text/descriptions.py: This will generate and store text descriptions of the images rendered using Blender
3) Run ./text/captions.py: This will simplify the generated descriptions into captions
4) Run ./text/fusion.py: This will fuse the simplified captions into a single descriptive caption
5) Run ./text/embeddings.py: This will generate text embeddings from captions and store them

### Tri-plane Ganeration and Fitting
1) Run triplane.py: This will generate and store tri-plane representations
2) Run create_voxel.py: This will generate and store voxel data of 3D models
3) Run mlp.py: This will train the mlp and save its weights

### Variational Auto-Encoder (VAE)
Run vae.py: This will train the VAE on the generated tri-planes and convert them into latent tri-planes

### Latent Diffusion Model (LDM)
Run ldm.py: This will train the ldm model on latent tri-planes and use the generated text embeddings to condition the model

### Generate Models
Run generator.py: This will generate and save 3D models

### Quantative Results
Run clip.py: This will compute the average CLIP cosine similarity score for all the generated models

## Datasets

### 3D Dataset
Dataset URL: [ShapeNet/ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)

### Captions Dataset (Generated as part of this project)
Dataset URL: [Rohan3/ShapeNetCore_Captions](https://huggingface.co/datasets/Rohan3/ShapeNetCore_Captions)