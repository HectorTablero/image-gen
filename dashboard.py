import toml
from image_gen.base import GenerativeModel
import streamlit as st
import torch
import io
from PIL import Image
import os
import time
import numpy as np
import base64
import sys
sys.path.append('./..')


def tensor_to_image(tensor):
    """Convert a tensor to PIL Image"""
    tensor = tensor.detach().cpu()

    # Handle different channel dimensions
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first sample

    # Normalize to [0, 1] if not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)

    # Convert to PIL Image
    if tensor.shape[-1] == 1:  # Grayscale
        return Image.fromarray(tensor[..., 0], mode='L')
    return Image.fromarray(tensor)


@st.cache_data
def load_css():
    with open("assets/styles.css", "r", encoding="utf-8") as f:
        return f.read()


about = [
    "This dashboard offers an interactive way to manage and utilize generative models for image generation, colorization, and imputation.",
    "**Authors:** Álvaro Martínez Gamo, Héctor Tablero Díaz",
    "",
    "_The python module and dashboard were made as a project for the subject Aprendizaje Automático 3 (Machine Learning 3) in the [Data Science and Engineering](https://www.uam.es/uam/en/ingenieria-datos) degree at the Autonomous University of Madrid._"
]


def add_additional_info():
    # Icono de https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded
    with st.expander("Dashboard Information", icon=":material/info:"):
        for i in about:
            st.write(i)


def model_selection():
    st.title("Model Management")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Load Model")
        uploaded_file = st.file_uploader(
            "Upload model checkpoint", type=["pt", "pth"])

        if 'model_dir' not in st.session_state:
            st.session_state.model_dir = "saved_models"
            os.makedirs(st.session_state.model_dir, exist_ok=True)

        if uploaded_file is not None:
            try:
                # Create a unique key for this upload session
                upload_key = f"uploaded_{uploaded_file.name}"

                # Only process if this is a new upload (not from rerun)
                if upload_key not in st.session_state:
                    st.session_state[upload_key] = True

                    save_path = os.path.join(
                        st.session_state.model_dir, uploaded_file.name)

                    # Save the file
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Verify and load the model
                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        model = GenerativeModel(verbose=False)
                        model.load(save_path)
                        st.session_state.model = model
                        st.session_state.current_model = uploaded_file.name

                    st.success(f"Model saved and loaded from {save_path}")

                    # Clear the uploader after successful processing
                    uploaded_file = None
                    time.sleep(1)  # Let user see success message
                    st.rerun()

            except Exception as e:
                st.error(f"Invalid model file: {str(e)}")
                # Clear the failed upload from session state
                if upload_key in st.session_state:
                    del st.session_state[upload_key]

        model_dir = st.text_input(
            "Model directory", value=st.session_state.model_dir)
        st.session_state.model_dir = model_dir

        if st.button("Refresh Model List"):
            pass  # Triggers rerun

        try:
            models = [f for f in os.listdir(
                model_dir) if f.endswith(".pt") or f.endswith(".pth")]
            selected_model = st.selectbox("Available models", models)

            if st.button("Load Selected Model"):
                model_path = os.path.join(model_dir, selected_model)
                try:
                    model = GenerativeModel(verbose=False)
                    model.load(model_path)
                    st.session_state.model = model
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

        except FileNotFoundError:
            st.error("Model directory not found")
            if st.button("Create directory"):
                os.makedirs(st.session_state.model_dir, exist_ok=True)
                st.rerun()

    with col2:
        st.header("Model Information")
        if st.session_state.model is not None:
            info = {
                "Number of Channels": st.session_state.model.num_c,
                "Input Shape": st.session_state.model.shape,
                "Sampler Type": type(st.session_state.model.sampler).__name__,
                "Diffusion Type": {
                    type(
                        st.session_state.model.diffusion).__name__: st.session_state.model.diffusion.config()
                }
            }
            if st.session_state.model.diffusion.NEEDS_NOISE_SCHEDULE:
                info["Noise Schedule"] = {
                    type(
                        st.session_state.model.diffusion.schedule).__name__: st.session_state.model.diffusion.schedule.config()
                }
            st.json(info)
        else:
            st.warning("No model loaded")

    add_additional_info()


def generation():
    st.title("Image Generation")

    if st.session_state.model is None:
        st.warning(
            "Please load a model first from Model Management")
        return

    with st.expander("Generation Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            num_images = st.slider("Number of images", 1, 16, 4)
            use_seed = st.checkbox("Use fixed seed", value=True)
            if use_seed:
                seed = st.number_input("Seed", value=42)
            else:
                seed = None
                st.number_input("Seed", value=42, disabled=True,
                                key="disabled_seed")

        with col2:
            steps = st.slider("Sampling steps", 10, 1000, 500)
            show_progress = st.checkbox("Show generation progress", True)

    if st.button("Generate Images"):
        try:
            placeholder = st.empty()
            progress_bars_created = False

            def get_columns_distribution(n_images):
                if n_images <= 5:
                    return [n_images]

                if n_images <= 16:
                    distributions = {
                        6: [3, 3],
                        7: [4, 3],
                        8: [4, 4],
                        9: [5, 4],
                        10: [5, 5],
                        11: [4, 4, 3],
                        12: [4, 4, 4],
                        13: [5, 4, 4],
                        14: [5, 5, 4],
                        15: [5, 5, 5],
                        16: [4, 4, 4, 4],
                    }
                    return distributions[n_images]

                ret = [5] * n_images // 5
                if n_images % 5 != 0:
                    ret.append(n_images % 5)
                return ret

            def update_progress(x_t, step):
                nonlocal progress_bars_created
                current_images = [tensor_to_image(img) for img in x_t]
                distribution = get_columns_distribution(len(current_images))

                with placeholder.container():
                    row_start = 0
                    for cols_in_row in distribution:
                        # Create columns for current row
                        cols = st.columns(cols_in_row)
                        images_in_row = current_images[row_start:row_start+cols_in_row]

                        for idx, (img, col) in enumerate(zip(images_in_row, cols)):
                            with col:
                                # Progress image display
                                buf = io.BytesIO()
                                img.save(buf, format="PNG")
                                img_bytes = base64.b64encode(
                                    buf.getvalue()).decode("utf-8")

                                html = f"""
                                <div class="image-container">
                                    <img src="data:image/png;base64,{img_bytes}" style="width: 100%; height: auto;"/>
                                    <div class="overlay">
                                        <div class="spinner"></div>
                                    </div>
                                </div>
                                """
                                st.markdown(html, unsafe_allow_html=True)

                                # Progress bars
                                if not progress_bars_created:
                                    st.session_state[f"pb_{row_start+idx}"] = st.progress(
                                        0)
                                else:
                                    st.session_state[f"pb_{row_start+idx}"].progress(
                                        (step + 1) / steps)

                        row_start += cols_in_row

                progress_bars_created = True

            # Generate images
            generated = st.session_state.model.generate(
                num_images,
                n_steps=steps,
                seed=seed if use_seed else None,
                progress_callback=update_progress if show_progress else None
            )

            # Final images display
            images = [tensor_to_image(img) for img in generated]
            distribution = get_columns_distribution(len(images))

            with placeholder.container():
                row_start = 0
                for cols_in_row in distribution:
                    # Create columns for current row
                    cols = st.columns(cols_in_row)
                    images_in_row = images[row_start:row_start+cols_in_row]

                    for idx, (img, col) in enumerate(zip(images_in_row, cols)):
                        with col:
                            # Final image display
                            st.image(img, use_container_width=True)

                            # Download button
                            buf = io.BytesIO()
                            img.save(buf, format="PNG", compress_level=0)
                            st.download_button(
                                f"Download Image {row_start+idx+1}",
                                buf.getvalue(),
                                f"generated_{row_start+idx+1}.png",
                                "image/png",
                                key=f"dl_{row_start+idx}",
                                on_click=lambda: None
                            )

                    row_start += cols_in_row

            # st.success(
            #     f"Generated {num_images} images in {time.time()-start_time:.2f}s")

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")


def colorization():
    st.title("Image Colorization")

    if st.session_state.model is None:
        st.warning(
            "Please load a model first from Model Management")
        return

    uploaded_file = st.file_uploader(
        "Upload grayscale image", type=["jpg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("L")

        with col1:
            st.image(image, caption="Original Grayscale",
                     use_container_width=True)

        if st.button("Colorize Image"):
            try:
                # Convert to tensor and process
                tensor = torch.tensor(np.array(image) / 255.0).unsqueeze(0)
                colored = st.session_state.model.colorize(tensor)
                colored_img = tensor_to_image(colored[0])

                with col2:
                    st.image(colored_img, caption="Colorized",
                             use_container_width=True)
                    buf = io.BytesIO()
                    colored_img.save(buf, format="PNG")
                    st.download_button(
                        "Download Colorized Image",
                        buf.getvalue(),
                        "colorized.png",
                        "image/png"
                    )

            except Exception as e:
                st.error(f"Colorization failed: {str(e)}")


def imputation():
    st.title("Image Imputation")

    if st.session_state.model is None:
        st.warning("Please load a model first from Model Management")
        return

    uploaded_file = st.file_uploader(
        "Upload image", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image",
                 use_container_width=True)


pages = {
    "Management": [
        st.Page(model_selection, title="Model Selection")
    ],
    "Generation": [
        st.Page(generation, title="Image Generation"),
        st.Page(colorization, title="Colorization"),
        st.Page(imputation, title="Imputation")
    ]
}


def main():
    st.set_page_config(page_title="Image Generation Dashboard", layout="wide", page_icon=":frame_with_picture:",
                       initial_sidebar_state="expanded", menu_items={"about": "\n\n".join(about) + "\n\n---"})

    primary_color = toml.load(
        ".streamlit/config.toml")["theme"]["primaryColor"]
    st.html("<style>" + load_css() + "\na {\n    color: " + primary_color + " !important;\n}\n\ncode {\n    color: " +
            primary_color + " !important;\n}\n\n:root {\n    --primary-color: " + primary_color + ";\n}</style>")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'current_model_info' not in st.session_state:
        st.session_state.current_model_info = None

    pg = st.navigation(pages, expanded=True)
    pg.run()


if __name__ == "__main__":
    main()
