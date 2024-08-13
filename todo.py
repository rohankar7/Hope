# TODO:
# The generated .obj files do not have faces in the form of 'f  7056//4987 7052//4987 7055//4987'
# Convert the saved triplanes to a latent triplane representation of shape (c, latent_dim, latent_dim)

# 3.2 Raw tri-plane fitting
#   Hybrid Neural SDF
#   Training points sampling: Sample points from the surface of .obj models and compute the SDF values
#   Triplane Fitting: Fit the triplanes to an MLP
# 3.3 Compressing to latent tri-plane space
# Use the latent triplanes to train the Diffusion model
# Use the Diffusion model to generate triplanes
# Use the triplanes generated from the diffusion model to generate the .obj model
# Convert text descriptions to latent space
# Use those latent space to map the text descriptions to 3D models
# Convert meshes to watertight meshes: https://chatgpt.com/share/56d27eb5-01aa-493a-80df-e48287f9721f
# Apply mesh refinement

# TODO:
# Text Embeddings
# Create a Denoising U-net
# Include Self-Attention blocks

# TODO:
# Apply for extension

# QUESTIONS:------------------------------------------------------------
# How to improve the VAE loss when it stops around 35000
# How to reshape the latents to a proper .obj model similar to the original model
# How convert the text descriptions of the 3D models to text embeddings and then convert those embeddings to the same latent space used to train the diffusion model
# Should we train the diffusion model on the latent triplane data or also on the data of the latent text embeddings
# How to extract features like colour and density from the .obj files