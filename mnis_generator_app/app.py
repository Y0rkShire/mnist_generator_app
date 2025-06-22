import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator_model import Generator

# Load Generator
device = "cpu"
model = Generator().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

# Generate digit images
def generate_images(num_images=5):
    z = torch.randn(num_images, 100).to(device)
    with torch.no_grad():
        imgs = model(z).cpu()
    return imgs

# Streamlit UI
st.title("MNIST Digit Generator")
st.write("This app generates handwritten-style digits using a GAN trained on MNIST.")

if st.button("Generate 5 Random Digits"):
    images = generate_images(5)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
