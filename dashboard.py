"""
Interactive dashboard for generative models with capabilities for image generation,
colorization, and imputation.

This module provides a Streamlit-based UI to interact with generative models,
offering visualization tools and user-friendly controls.
"""

# Standard library imports
from image_gen.samplers import (EulerMaruyama, ExponentialIntegrator,
                                ODEProbabilityFlow, PredictorCorrector)
from image_gen.base import GenerativeModel
import base64
import io
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import streamlit as st
import toml
import torch
from PIL import Image

# Local application imports
sys.path.append('./..')

# Constants
NONE_LABEL = "(unset)"


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL Image.

    Args:
        tensor: PyTorch tensor representing an image.

    Returns:
        PIL Image converted from the tensor.
    """
    tensor = tensor.detach().cpu()

    if tensor.dim() == 4:
        tensor = tensor[0]

    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    if tensor.shape[0] == 1:  # Grayscale
        tensor = tensor.squeeze(0)
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode='L')
    else:  # RGB
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
        array = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)


@st.cache_data
def load_css() -> str:
    """
    Load CSS styles from file.

    Returns:
        String containing CSS styles.
    """
    with open("assets/styles.css", "r", encoding="utf-8") as f:
        return f.read()


# Dashboard information text
ABOUT = [
    "This dashboard offers an interactive way to manage and utilize generative "
    "models for image generation, colorization, and imputation.",
    "**Authors:** Álvaro Martínez Gamo, Héctor Tablero Díaz",
    "",
    "_The python module and dashboard were made as a project for the subject "
    "Aprendizaje Automático 3 (Machine Learning 3) in the "
    "[Data Science and Engineering](https://www.uam.es/uam/en/ingenieria-datos) "
    "degree at the Autonomous University of Madrid._"
]


def add_additional_info() -> None:
    """Add dashboard information in an expandable section."""
    # Icon from https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded
    with st.expander("Dashboard Information", icon=":material/info:"):
        for info_line in ABOUT:
            st.write(info_line)


def _get_class_name(obj: type) -> str:
    """
    Get the class name of an object.

    Args:
        obj: Object to get class name from.

    Returns:
        Class name as a string.
    """
    if hasattr(obj, "_class_name"):
        return obj._class_name
    return obj.__class__.__name__


def load_model_with_safety_check(model_path: str, display_name: str) -> Optional[GenerativeModel]:
    """
    Load a model with safety checks for custom classes.

    Args:
        model_path: Path to the model file.
        display_name: Display name for the model (for UI).

    Returns:
        Loaded model or None if loading was canceled.
    """
    # First, check if the model contains custom classes without loading them
    custom_classes = {}

    try:
        # Load the model state to examine it
        state_dict = torch.load(model_path, map_location='cpu')

        # Check diffusion type
        if '_type' in state_dict.get('diffusion', {}):
            diffusion_type = state_dict['diffusion']['_type']
            if diffusion_type not in ['GaussianDiffusion', 'ImprovedDDPM']:
                custom_classes['diffusion'] = {
                    'name': diffusion_type,
                    'code': state_dict['diffusion'].get('_code', 'Code not available')
                }

        # Check sampler type
        if '_type' in state_dict.get('sampler', {}):
            sampler_type = state_dict['sampler']['_type']
            if sampler_type not in ['EulerMaruyama', 'ExponentialIntegrator',
                                    'ODEProbabilityFlow', 'PredictorCorrector']:
                custom_classes['sampler'] = {
                    'name': sampler_type,
                    'code': state_dict['sampler'].get('_code', 'Code not available')
                }

        # Check noise schedule type
        if '_type' in state_dict.get('noise_schedule', {}):
            schedule_type = state_dict['noise_schedule']['_type']
            if schedule_type not in ['Linear', 'Cosine']:
                custom_classes['noise_schedule'] = {
                    'name': schedule_type,
                    'code': state_dict['noise_schedule'].get('_code', 'Code not available')
                }

        # Check for other custom components
        for key, value in state_dict.items():
            if isinstance(value, dict) and '_type' in value:
                if key not in ['diffusion', 'sampler', 'noise_schedule']:
                    custom_classes[key] = {
                        'name': value['_type'],
                        'code': value.get('_code', 'Code not available')
                    }

    except Exception as e:
        st.error(f"Error examining model file: {str(e)}")
        return None

    # If custom classes are found, show the safety dialog
    if custom_classes:
        # Show warning dialog
        with st.dialog("⚠️ Custom Classes Detected", can_be_closed=False):
            st.markdown("""
            ## ⚠️ Security Warning

            This model contains custom classes that could potentially execute harmful code 
            when loaded. Please review the code below before proceeding.

            **Never load models from untrusted sources without reviewing their code.**
            """)

            # Add accordions for each custom class
            for class_type, class_info in custom_classes.items():
                with st.expander(f"Custom {class_type.title()}: {class_info['name']}"):
                    st.code(class_info['code'], language="python")

            col1, col2 = st.columns([3, 1])

            # Create a placeholder for the countdown button
            load_btn_placeholder = col1.empty()

            # Initialize countdown timer
            countdown = 3

            # Disable button for 3 seconds with countdown
            for i in range(countdown, 0, -1):
                load_btn_placeholder.button(
                    f"Load Anyway ({i})",
                    disabled=True,
                    type="primary",
                    use_container_width=True
                )
                time.sleep(1)

            # Create the enabled button after countdown
            load_confirmed = load_btn_placeholder.button(
                "Load Anyway",
                type="primary",
                use_container_width=True
            )

            cancel = col2.button("Cancel", use_container_width=True)

            if cancel:
                st.rerun()
                return None

            if not load_confirmed:
                # Button not clicked yet, keep dialog open
                st.stop()

    # If we reach here, either there are no custom classes or the user confirmed loading
    try:
        with st.spinner(f"Loading {display_name}..."):
            model = GenerativeModel(verbose=False)
            model.load(model_path, unsafe=True)
            st.success(f"Model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def model_selection() -> None:
    """
    Render the model management page.

    Provides UI for loading, selecting, and displaying model information.
    """
    st.title("Model Management")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Load Model")
        uploaded_file = st.file_uploader(
            "Upload model checkpoint · Will be copied into the model directory",
            type=["pt", "pth"]
        )

        if 'model_dir' not in st.session_state:
            st.session_state.model_dir = "examples/saved_models"

        if uploaded_file is not None:
            try:
                # Create a unique key for this upload session
                upload_key = f"uploaded_{uploaded_file.name}"

                # Only process if this is a new upload (not from rerun)
                if upload_key not in st.session_state:
                    st.session_state[upload_key] = True

                    save_path = os.path.join(
                        st.session_state.model_dir, uploaded_file.name
                    )

                    # Save the file
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load the model with safety check
                    model = load_model_with_safety_check(
                        save_path, uploaded_file.name)

                    if model is not None:
                        st.session_state.model = model
                        st.session_state.model_name = ".".join(
                            uploaded_file.name.split(".")[:-1]
                        )
                        st.session_state.current_model = uploaded_file.name

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
            "Model directory", value=st.session_state.model_dir
        )

        if model_dir != st.session_state.model_dir:
            st.session_state.model_dir = model_dir

        if st.button("Refresh Model List", icon=":material/sync:"):
            pass  # Triggers rerun

        try:
            models = [
                f for f in os.listdir(model_dir)
                if f.endswith(".pt") or f.endswith(".pth")
            ]
            selected_model = st.selectbox("Available models", models)

            if st.button("Load Selected Model", icon=":material/check:"):
                model_path = os.path.join(model_dir, selected_model)

                # Load model with safety check
                model = load_model_with_safety_check(
                    model_path, selected_model)

                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_name = ".".join(
                        selected_model.split(".")[:-1]
                    )
                    st.session_state.previous_sampler = None
                    st.rerun()

        except FileNotFoundError:
            st.error("Model directory not found")
            if st.button("Create directory"):
                os.makedirs(st.session_state.model_dir, exist_ok=True)
                st.rerun()

        if st.session_state.model is not None:
            st.success("Model loaded successfully!")

    with col2:
        st.header("Model Information")
        if st.session_state.model is not None:
            sampler_options = {
                "Euler-Maruyama": "em",
                "Exponential Integrator": "exp",
                "ODE Probability Flow": "ode",
                "Predictor-Corrector": "pred"
            }

            sampler_classes = [
                "EulerMaruyama", "ExponentialIntegrator",
                "ODEProbabilityFlow", "PredictorCorrector"
            ]

            current_sampler = _get_class_name(st.session_state.model.sampler)

            index = (
                sampler_classes.index(current_sampler)
                if current_sampler in sampler_classes else 4
            )
            if index == 4:
                sampler_options["Custom"] = "custom"

            # Create select box with current sampler as default
            selected_sampler = st.selectbox(
                "Sampler Type",
                options=list(sampler_options.keys()),
                index=index,
                key="sampler_select"
            )

            # Automatically update sampler when selection changes
            if st.session_state.get("previous_sampler") != selected_sampler:
                try:
                    # Get selected sampler class
                    sampler_cls = sampler_options[selected_sampler]

                    # Reinitialize sampler with model's diffusion
                    st.session_state.model.sampler = sampler_cls
                    if st.session_state.previous_sampler is not None:
                        st.toast(
                            f"Sampler changed to {selected_sampler}", icon="🔄"
                        )
                    st.session_state.previous_sampler = selected_sampler
                except Exception as e:
                    st.error(f"Failed to change sampler: {str(e)}")

            info = {
                "Model Name": st.session_state.model_name,
                "Number of Channels": st.session_state.model.num_channels,
                "Input Shape": st.session_state.model.shape,
                "Sampler Type": _get_class_name(st.session_state.model.sampler),
                "Diffusion Type": {
                    _get_class_name(st.session_state.model.diffusion):
                        st.session_state.model.diffusion.config()
                }
            }

            if st.session_state.model.diffusion.NEEDS_NOISE_SCHEDULE:
                info["Noise Schedule"] = {
                    _get_class_name(st.session_state.model.diffusion.schedule):
                        st.session_state.model.diffusion.schedule.config()
                }

            if st.session_state.model._label_map is not None:
                info["Labels"] = ", ".join(
                    [str(i) for i in st.session_state.model._label_map.keys()]
                )

            info["Model Version"] = st.session_state.model.version
            st.json(info)
        else:
            st.warning("No model loaded")

    add_additional_info()


def generation() -> None:
    """
    Render the image generation page.

    Provides UI for generating images with the loaded model.
    """
    st.title("Image Generation")

    if st.session_state.model is None:
        st.warning("Please load a model first from Model Management")
        return

    # Get label map from model data
    label_map = st.session_state.model._label_map

    with st.expander(
        "Generation Settings", expanded=True, icon=":material/tune:"
    ):
        col1, col2 = st.columns(2)

        with col1:
            num_images = st.slider(
                "Number of images", 1, 16,
                st.session_state.settings["num_images"]
            )
            st.session_state.settings["num_images"] = num_images
            use_seed = st.checkbox(
                "Use fixed seed", st.session_state.settings["use_seed"]
            )
            st.session_state.settings["use_seed"] = use_seed
            if use_seed:
                seed = st.number_input(
                    "Seed", st.session_state.settings["seed"]
                )
                st.session_state.settings["seed"] = seed
            else:
                seed = None
                st.number_input(
                    "Seed", st.session_state.settings["seed"],
                    disabled=True, key="generation_seed"
                )

        with col2:
            steps = st.slider(
                "Sampling steps", 10, 1000,
                st.session_state.settings["steps"]
            )
            st.session_state.settings["steps"] = steps
            show_progress = st.checkbox(
                "Show generation progress",
                st.session_state.settings["show_progress"]
            )
            st.session_state.settings["show_progress"] = show_progress

            # Class selection only if label map exists
            if label_map is not None:
                selected_class = st.selectbox(
                    "Class conditioning",
                    options=[NONE_LABEL] + sorted(label_map.keys()),
                    index=(
                        list(label_map.keys()).index(
                            st.session_state.settings["class_label"]
                        ) if st.session_state.settings["class_label"] in label_map
                        else 0
                    )
                )
                class_id = (
                    label_map[selected_class] if selected_class != NONE_LABEL
                    else None
                )
                st.session_state.settings["class_id"] = selected_class
            else:
                class_id = None

    # brush / gradient / auto_awesome
    if st.button("Generate Images", icon=":material/auto_awesome:"):
        try:
            placeholder = st.empty()
            progress_bars_created = False

            def get_columns_distribution(n_images: int) -> List[int]:
                """
                Determine the optimal column distribution for displaying images.

                Args:
                    n_images: Number of images to distribute.

                Returns:
                    List of integers representing number of columns per row.
                """
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

                ret = [5] * (n_images // 5)
                if n_images % 5 != 0:
                    ret.append(n_images % 5)
                return ret

            def update_progress(
                x_t: torch.Tensor, step: int
            ) -> None:
                """
                Update progress display during image generation.

                Args:
                    x_t: Current image tensor.
                    step: Current generation step.
                """
                nonlocal progress_bars_created
                current_images = [tensor_to_image(img) for img in x_t]
                distribution = get_columns_distribution(len(current_images))

                with placeholder.container():
                    row_start = 0
                    for cols_in_row in distribution:
                        cols = st.columns(cols_in_row)
                        images_in_row = current_images[
                            row_start:row_start+cols_in_row
                        ]

                        for idx, (img, col) in enumerate(
                            zip(images_in_row, cols)
                        ):
                            with col:
                                # Progress image display
                                buf = io.BytesIO()
                                img.save(buf, format="PNG")
                                img_bytes = base64.b64encode(
                                    buf.getvalue()
                                ).decode("utf-8")

                                html = f"""
                                <div class="image-container">
                                    <img src="data:image/png;base64,{img_bytes}" 
                                         style="width: 100%; height: auto;"/>
                                    <div class="overlay">
                                        <div class="spinner"></div>
                                    </div>
                                </div>
                                """
                                st.markdown(html, unsafe_allow_html=True)

                                # Progress bars
                                pb_key = f"pb_{row_start+idx}"
                                if not progress_bars_created:
                                    st.session_state[pb_key] = st.progress(0)
                                else:
                                    st.session_state[pb_key].progress(
                                        (step + 1) / steps
                                    )

                        row_start += cols_in_row

                progress_bars_created = True

            # Generate images
            generated = st.session_state.model.generate(
                num_images,
                n_steps=steps,
                seed=seed if use_seed else None,
                class_labels=class_id,
                progress_callback=(
                    update_progress if show_progress else None
                )
            )

            # Final images display
            images = [tensor_to_image(img) for img in generated]
            distribution = get_columns_distribution(len(images))

            with placeholder.container():
                row_start = 0
                for cols_in_row in distribution:
                    cols = st.columns(cols_in_row)
                    images_in_row = images[row_start:row_start+cols_in_row]

                    for idx, (img, col) in enumerate(
                        zip(images_in_row, cols)
                    ):
                        with col:
                            st.image(img, use_container_width=True)

                            @st.fragment
                            def download_image(buf: io.BytesIO, n: int) -> None:
                                """
                                Create download button for an image.

                                Args:
                                    buf: BytesIO object with image data.
                                    n: Image number for filename.
                                """
                                st.download_button(
                                    f"Download Image {n}",
                                    buf.getvalue(),
                                    f"generated_{n}.png",
                                    "image/png",
                                    icon=":material/download:",
                                    key=f"dl_{n}",
                                    on_click=lambda: None
                                )

                            # Download button
                            buf = io.BytesIO()
                            img.save(buf, format="PNG", compress_level=0)
                            download_image(buf, row_start+idx+1)

                    row_start += cols_in_row

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")


def colorization() -> None:
    """
    Render the image colorization page.

    Provides UI for colorizing grayscale images.
    """
    st.title("Image Colorization")

    if st.session_state.model is None:
        st.warning("Please load a model first from Model Management")
        return

    # Get label map from model data
    label_map = st.session_state.model._label_map

    # Settings at the top
    with st.expander(
        "Colorization Settings", expanded=True, icon=":material/tune:"
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div style='height: 100%;'></div>",
                        unsafe_allow_html=True)
            use_seed = st.checkbox(
                "Use fixed seed", st.session_state.settings["use_seed"]
            )
            st.session_state.settings["use_seed"] = use_seed
            if use_seed:
                seed = st.number_input(
                    "Seed", st.session_state.settings["seed"]
                )
                st.session_state.settings["seed"] = seed
            else:
                seed = None
                st.number_input(
                    "Seed", st.session_state.settings["seed"],
                    disabled=True, key="colorization_seed"
                )

        with col2:
            steps = st.slider(
                "Sampling steps", 10, 1000,
                st.session_state.settings["steps"]
            )
            st.session_state.settings["steps"] = steps
            show_progress = st.checkbox(
                "Show generation progress",
                st.session_state.settings["show_progress"]
            )
            st.session_state.settings["show_progress"] = show_progress

            # Class selection only if label map exists
            if label_map is not None:
                selected_class = st.selectbox(
                    "Class conditioning",
                    options=[NONE_LABEL] + sorted(label_map.keys()),
                    index=(
                        list(label_map.keys()).index(
                            st.session_state.settings["class_label"]
                        ) if st.session_state.settings["class_label"] in label_map
                        else 0
                    )
                )
                class_id = (
                    label_map[selected_class] if selected_class != NONE_LABEL
                    else None
                )
                st.session_state.settings["class_id"] = selected_class
            else:
                class_id = None

    uploaded_file = st.file_uploader(
        "Upload grayscale image", type=["jpg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("L")

        with col1:
            st.image(
                image, caption="Original Grayscale",
                use_container_width=True
            )

            if st.button("Colorize Image", icon=":material/auto_awesome:"):
                try:
                    placeholder = col2.empty()
                    progress_bar = None

                    def update_progress(
                        current_tensor: torch.Tensor, step: int
                    ) -> None:
                        """
                        Update progress display during colorization.

                        Args:
                            current_tensor: Current image tensor.
                            step: Current colorization step.
                        """
                        nonlocal progress_bar
                        current_img = tensor_to_image(current_tensor[0])

                        with placeholder.container():
                            # Progress image display with overlay
                            buf = io.BytesIO()
                            current_img.save(buf, format="PNG")
                            img_bytes = base64.b64encode(
                                buf.getvalue()
                            ).decode("utf-8")

                            html = f"""
                            <div class="image-container">
                                <img src="data:image/png;base64,{img_bytes}" 
                                     style="width: 100%; height: auto;"/>
                                <div class="overlay">
                                    <div class="spinner"></div>
                                </div>
                            </div>
                            """
                            st.markdown(html, unsafe_allow_html=True)

                            # Progress bar
                            if not progress_bar:
                                progress_bar = st.progress(0)
                            progress_bar.progress((step + 1) / steps)

                    # Convert to tensor and process
                    tensor = torch.tensor(
                        np.array(image) / 255.0
                    ).unsqueeze(0)
                    colored = st.session_state.model.colorize(
                        tensor,
                        n_steps=steps,
                        seed=seed if use_seed else None,
                        class_labels=class_id,
                        progress_callback=(
                            update_progress if show_progress else None
                        )
                    )
                    colored_img = tensor_to_image(colored[0])

                    # Final display
                    with placeholder.container():
                        st.image(
                            colored_img, caption="Colorized Result",
                            use_container_width=True
                        )

                        # Download button
                        buf = io.BytesIO()
                        colored_img.save(buf, format="PNG")

                        @st.fragment
                        def create_download() -> None:
                            """Create download button for the colorized image."""
                            st.download_button(
                                "Download Colorized Image",
                                buf.getvalue(),
                                "colorized.png",
                                "image/png",
                                icon=":material/download:",
                                key="colorized_dl"
                            )
                        create_download()

                except Exception as e:
                    st.error(f"Colorization failed: {str(e)}")


def imputation() -> None:
    """
    Render the image imputation page.

    Provides UI for filling in transparent areas of images.
    """
    st.title("Image Imputation")

    if st.session_state.model is None:
        st.warning("Please load a model first from Model Management")
        return

    # Get label map from model data
    label_map = st.session_state.model._label_map

    # Settings at the top
    with st.expander(
        "Imputation Settings", expanded=True, icon=":material/tune:"
    ):
        col1, col2 = st.columns(2)

        with col1:
            use_seed = st.checkbox(
                "Use fixed seed", st.session_state.settings["use_seed"]
            )
            st.session_state.settings["use_seed"] = use_seed
            if use_seed:
                seed = st.number_input(
                    "Seed", st.session_state.settings["seed"]
                )
                st.session_state.settings["seed"] = seed
            else:
                seed = None
                st.number_input(
                    "Seed", st.session_state.settings["seed"],
                    disabled=True, key="imputation_seed"
                )

        with col2:
            steps = st.slider(
                "Sampling steps", 10, 1000,
                st.session_state.settings["steps"]
            )
            st.session_state.settings["steps"] = steps
            show_progress = st.checkbox(
                "Show generation progress",
                st.session_state.settings["show_progress"]
            )
            st.session_state.settings["show_progress"] = show_progress

            # Class selection only if label map exists
            if label_map is not None:
                selected_class = st.selectbox(
                    "Class conditioning",
                    options=[NONE_LABEL] + sorted(label_map.keys()),
                    index=(
                        list(label_map.keys()).index(
                            st.session_state.settings["class_label"]
                        ) if st.session_state.settings["class_label"] in label_map
                        else 0
                    )
                )
                class_id = (
                    label_map[selected_class] if selected_class != NONE_LABEL
                    else None
                )
                st.session_state.settings["class_label"] = selected_class
            else:
                class_id = None

    uploaded_file = st.file_uploader(
        "Upload image with transparent areas to inpaint",
        type=["png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGBA")  # Force RGBA mode
        width, height = image.size

        # Split into RGB and alpha channels
        rgb_img = image.convert("RGB")
        alpha_channel = np.array(image.split()[-1])

        # Create mask from transparency (1 = masked/inpaint area)
        mask = (alpha_channel == 0)

        if not np.any(mask):
            st.warning(
                "No transparent areas detected - please upload an image "
                "with transparent regions to inpaint"
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            st.markdown(f"""
                <div class="image-mask-container">
                    <div class="checkerboard-bg">
                        <img class="imputation-image" style="image-rendering: pixelated;" 
                             src="data:image/png;base64,{img_b64}" />
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "<p style='text-align: center;'>Original Image with "
                "Transparency</p>",
                unsafe_allow_html=True
            )

            if st.button("Impute Image", icon=":material/auto_awesome:"):
                try:
                    device = st.session_state.model.device

                    # Convert to tensors and move to model device
                    img_tensor = torch.tensor(
                        np.array(rgb_img)/255.0
                    ).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    mask_tensor = torch.from_numpy(mask).unsqueeze(
                        0
                    ).unsqueeze(0).to(device)

                    # Setup progress display
                    placeholder = col2.empty()
                    progress_bar = None

                    def update_progress(
                        current_tensor: torch.Tensor, step: int
                    ) -> None:
                        """
                        Update progress display during imputation.

                        Args:
                            current_tensor: Current image tensor.
                            step: Current imputation step.
                        """
                        nonlocal progress_bar
                        current_img = tensor_to_image(current_tensor[0].cpu())

                        with placeholder.container():
                            # Progress image display with overlay
                            buf = io.BytesIO()
                            current_img.save(buf, format="PNG")
                            img_bytes = base64.b64encode(
                                buf.getvalue()
                            ).decode("utf-8")

                            html = f"""
                            <div class="image-container">
                                <img src="data:image/png;base64,{img_bytes}" 
                                     style="width: 100%; height: auto;"/>
                                <div class="overlay">
                                    <div class="spinner"></div>
                                </div>
                            </div>
                            """
                            st.markdown(html, unsafe_allow_html=True)

                            if not progress_bar:
                                progress_bar = st.progress(0)
                            progress_bar.progress((step + 1) / steps)

                    # Run imputation
                    imputed = st.session_state.model.imputation(
                        x=img_tensor,
                        mask=mask_tensor,
                        n_steps=steps,
                        seed=seed if use_seed else None,
                        class_labels=class_id,
                        progress_callback=(
                            update_progress if show_progress else None
                        )
                    )

                    # Convert result back to CPU for display
                    imputed_img = tensor_to_image(imputed[0].cpu())

                    # Final display
                    with placeholder.container():
                        st.image(
                            imputed_img, caption="Imputed Result",
                            use_container_width=True
                        )

                        buf = io.BytesIO()
                        imputed_img.save(buf, format="PNG")

                        @st.fragment
                        def create_download() -> None:
                            """Create download button for the imputed image."""
                            st.download_button(
                                "Download Imputed Image",
                                buf.getvalue(),
                                "imputed.png",
                                "image/png",
                                icon=":material/download:",
                                key="imputed_dl"
                            )
                        create_download()

                except Exception as e:
                    st.error(f"Imputation failed: {str(e)}")


# Define page hierarchy
pages = {
    "Management": [
        st.Page(model_selection, title="Model Selection",
                icon=":material/folder:"),
    ],
    "Generation": [
        st.Page(generation, title="Image Generation", icon=":material/image:"),
        st.Page(colorization, title="Colorization", icon=":material/palette:"),
        st.Page(imputation, title="Imputation", icon=":material/draw:"),
    ]
}


def main() -> None:
    """
    Initialize and run the Streamlit dashboard.

    Sets up the page configuration, styles, and session state variables.
    """
    st.set_page_config(
        page_title="Image Generation Dashboard",
        layout="wide",
        page_icon=":frame_with_picture:",
        initial_sidebar_state="expanded",
        menu_items={"about": "\n\n".join(ABOUT) + "\n\n---"}
    )

    primary_color = toml.load(
        ".streamlit/config.toml"
    )["theme"]["primaryColor"]

    st.html(
        "<style>" + load_css() +
        "\na {\n    color: " + primary_color + " !important;\n}\n\n" +
        "code {\n    color: " + primary_color + " !important;\n}\n\n" +
        ":root {\n    --primary-color: " + primary_color + ";\n}</style>"
    )

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    else:
        if (st.session_state.model is not None and
                st.session_state.model._label_map is None and
                len(st.session_state.model.stored_labels) > 1):
            st.session_state.model.set_labels(
                st.session_state.model.stored_labels
            )

    if 'model_name' not in st.session_state:
        st.session_state.model_name = None

    if 'model_dir' not in st.session_state:
        st.session_state.model_dir = "examples/saved_models"

    if 'current_model_info' not in st.session_state:
        st.session_state.current_model_info = None

    if 'settings' not in st.session_state:
        st.session_state.settings = {
            "num_images": 4,
            "show_progress": True,
            "use_seed": True,
            "seed": 42,
            "steps": 500,
            "class_label": NONE_LABEL
        }

    if st.session_state.model is None:
        st.html(f"""
            <style>
            [data-testid="stSidebarNavLink"] {{
                pointer-events: none;
                cursor: default;
                opacity: 0.5;
                color: #999 !important;
            }}
            [data-testid="stNavSectionHeader"]:first-child + li 
            a[data-testid="stSidebarNavLink"] {{
                pointer-events: auto;
                cursor: pointer;
                opacity: 1;
                color: inherit !important;
            }}
            </style>
        """)

    pg = st.navigation(pages, expanded=True)
    pg.run()


if __name__ == "__main__":
    main()
