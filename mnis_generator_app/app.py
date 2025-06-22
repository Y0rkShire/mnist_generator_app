import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator_model import Generator  # updated Generator
import numpy as np

# Load model
device = "cpu"
model = Generator().to(device)
model.load_state_dict(torch.load("generator_cgan.pth", map_location=device))
model.eval()

# UI
st.title("Conditional MNIST Digit Generator")
digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate 5 Images"):
    z = torch.randn(5, 100).to(device)
    labels = torch.tensor([digit] * 5).to(device)
    with torch.no_grad():
        images = model(z, labels).cpu()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
