import trimesh

file_path = 'C:/SHapeNetCore/02691156/10155655850468db78d106ce0a280f87/models/model_normalized.obj'
# file_path = './demo_model.obj'
mesh = trimesh.load(file_path, force='mesh')
mesh.show()