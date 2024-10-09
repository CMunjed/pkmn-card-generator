import gradio as gr
import numpy as np
import torch
import generator
from PIL import Image


MODEL_PATH = "models/model6-25-0.pt"
ngpu = 1
nz = 100  # Size of z latent vector (i.e. size of generator input).
nc = 3  # Number of channels in the images. For color images this is 3.
device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
)  # Load GPU.


def load_generator():
    gen = generator.Generator(ngpu, nz, nc)
    gen.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
    )
    gen.to(device)  # Move the model to the appropriate device (CPU or GPU)
    gen.eval()  # Set to evaluation mode

    return gen


def generate_card():
    gen = load_generator()

    # Generate a single latent vector
    single_noise = torch.randn(1, nz, 1, 1, device=device)

    # Generate a single image using the trained generator

    with torch.no_grad():  # Disable gradient calculation for inference
        generated_image = gen(single_noise).detach().cpu()

    # Assuming the generated image tensor is in the shape [1, 3, 256, 192]
    generated_image = generated_image.squeeze(0)  # Remove the batch dimension
    generated_image = (generated_image + 1) / 2  # Denormalize to the range [0, 1]

    # Convert the tensor to a numpy array
    generated_image_np = generated_image.permute(
        1, 2, 0
    ).numpy()  # Change the shape to [256, 192, 3]

    # Display the image using matplotlib
    # plt.imshow(generated_image_np)
    # plt.axis('off')  # Turn off the axis
    # plt.show()

    generated_image_np = (generated_image_np * 255).astype(np.uint8)

    generated_image_pil = Image.fromarray(generated_image_np)

    return generated_image_pil


# Gradio interface setup
demo = gr.Interface(
    fn=generate_card,
    inputs=None,
    outputs="image",
    title="Pokémon Card Generator",
    description="Click the button to generate a Pokémon card!",
)

# Launch the interface
demo.launch()
