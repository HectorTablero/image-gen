"""
Interactive dashboard for generative models with capabilities for image generation,
colorization, and imputation.

This module provides a Streamlit-based UI to interact with generative models,
offering visualization tools and user-friendly controls.
"""

# Standard library imports
from image_gen.base import GenerativeModel
import base64
import io
import os
import time
import json
from typing import Dict, List

# Third-party imports
import numpy as np
import streamlit as st
import toml
import torch
from PIL import Image

# debug
from pathlib import Path


def get_directory_tree(start_path="."):
    """Returns a string showing the directory tree structure"""
    result = []
    start_path = Path(start_path).resolve()

    for root, dirs, files in sorted(os.walk(start_path)):
        # Calculate indentation level
        level = len(Path(root).relative_to(start_path).parts)
        indent = '  ' * level

        # Add directory to result
        dir_name = os.path.basename(root)
        if level > 0:  # Skip the root directory name if we're starting from current dir
            result.append(f"{indent}{dir_name}/")

        # Add files to result
        for f in sorted(files):
            result.append(f"{indent}  {f}")

    return '\n'.join(result)

# # Local application imports
# sys.path.append('./..')


# Load translations from JSON files in the dashboard/locales directory
TRANSLATIONS = {}
AVAILABLE_LANGUAGES = ["en", "es", "cn"]
locales_dir = "dashboard/locales"

if os.path.exists(locales_dir):
    for filename in os.listdir(locales_dir):
        if filename.endswith(".json"):
            lang_code = os.path.splitext(filename)[0]
            with open(os.path.join(locales_dir, filename), "r", encoding="utf-8") as f:
                TRANSLATIONS[lang_code] = json.load(f)


def t(text: str) -> str:
    try:
        translation = TRANSLATIONS[st.session_state.get('language', 'en')]
        for key in text.split('.'):
            translation = translation[key]
        return translation
    except KeyError:
        return text


# Constants
NONE_LABEL = t("none_label")


DATASET_NAMES = {
    "mnist": t("datasets.mnist"),
    "cifar10": t("datasets.cifar10"),
}

DIFFUSER_NAMES = {
    "VarianceExploding": t("diffusers.ve"),
    "VariancePreserving": t("diffusers.vp"),
    "SubVariancePreserving": t("diffusers.svp")
}

NOISE_NAMES = {
    "LinearNoiseSchedule": t("noise.linear"),
    "CosineNoiseSchedule": t("noise.cosine")
}

SAMPLER_NAMES = {
    "EulerMaruyama": t("samplers.em"),
    "ExponentialIntegrator": t("samplers.exp"),
    "ODEProbabilityFlow": t("samplers.ode"),
    "PredictorCorrector": t("samplers.pred")
}


def _get_formatted_name(name: str, mapping: dict) -> str:
    return mapping.get(name, name)


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
    with open("dashboard/assets/styles.css", "r", encoding="utf-8") as f:
        return f.read()


# Dashboard information text
ABOUT = [
    t("about.1"),
    t("about.2"),
    "",
    t("about.3")
]


def add_additional_info() -> None:
    """Add dashboard information in an expandable section."""
    # Icon from https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded
    with st.expander(t("about.info"), icon=":material/info:"):
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


def model_selection_v1() -> None:
    """
    Render the model management page.

    Provides UI for loading, selecting, and displaying model information.
    """
    st.title(t("selection.title"))

    col1, col2 = st.columns(2)

    if st.session_state.get("previous_sampler") is None:
        st.session_state.previous_sampler = None

    with col1:
        st.header(t("selection.load"))
        uploaded_file = st.file_uploader(
            t("selection.upload"),
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

                    # Verify and load the model
                    with st.spinner(t("selection.loading").format(uploaded_file.name)):
                        model = GenerativeModel(verbose=False)
                        model.load(save_path, unsafe=True)
                        st.session_state.model = model
                        st.session_state.model_name = ".".join(
                            uploaded_file.name.split(".")[:-1])
                        st.session_state.current_model = uploaded_file.name

                    st.success(t("selection.loaded").format(save_path))

                    # Clear the uploader after successful processing
                    uploaded_file = None
                    time.sleep(1)  # Let user see success message
                    st.rerun()

            except Exception as e:
                st.error(t("selection.invalid").format(str(e)))
                # Clear the failed upload from session state
                if upload_key in st.session_state:
                    del st.session_state[upload_key]

        model_dir = st.text_input(
            t("selection.model_dir"), value=st.session_state.model_dir
        )

        if model_dir != st.session_state.model_dir:
            st.session_state.model_dir = model_dir

        if st.button(t("selection.refresh"), icon=":material/sync:"):
            pass  # Triggers rerun

        try:
            models = [
                f for f in os.listdir(model_dir)
                if f.endswith(".pt") or f.endswith(".pth")
            ]
            selected_model = st.selectbox(t("selection.available"), models)

            if st.button(t("selection.load_selected"), icon=":material/check:"):
                model_path = os.path.join(model_dir, selected_model)
                try:
                    model = GenerativeModel(verbose=False)
                    model.load(model_path, unsafe=True)
                    st.session_state.model = model
                    st.session_state.model_name = ".".join(
                        selected_model.split(".")[:-1])
                    st.session_state.previous_sampler = None
                    st.rerun()
                except Exception as e:
                    st.error(t("selection.error").format(str(e)))

        except FileNotFoundError:
            st.error(t("selection.dir_not_found"))
            if st.button(t("selection.create_dir"), icon=":material/folder-plus:"):
                os.makedirs(st.session_state.model_dir, exist_ok=True)
                st.rerun()

        if st.session_state.model is not None:
            st.success(t("selection.loaded_success"))

    with col2:
        st.header(t("selection.info"))
        if st.session_state.model is not None:
            sampler_options = {
                t("samplers.em"): "em",
                t("samplers.exp"): "exp",
                t("samplers.ode"): "ode",
                t("samplers.pred"): "pred"
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
                t("selection.sampler_type"),
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
                            t("selection.sampler_changed").format(selected_sampler), icon="🔄"
                        )
                    st.session_state.previous_sampler = selected_sampler
                except Exception as e:
                    st.error(t("selection.sampler_failed").format(str(e)))

            info = {
                t("selection.model_info.name"): st.session_state.model_name,
                t("selection.model_info.num_channels"): st.session_state.model.num_channels,
                t("selection.model_info.shape"): st.session_state.model.shape,
                t("selection.model_info.sampler"): _get_class_name(st.session_state.model.sampler),
                t("selection.model_info.diffusion"): {
                    _get_class_name(st.session_state.model.diffusion):
                        st.session_state.model.diffusion.config()
                }
            }

            if st.session_state.model.diffusion.NEEDS_NOISE_SCHEDULE:
                info[t("selection.model_info.noise")] = {
                    _get_class_name(st.session_state.model.diffusion.schedule):
                        st.session_state.model.diffusion.schedule.config()
                }

            if st.session_state.model._label_map is not None:
                info[t("selection.model_info.labels")] = ", ".join(
                    [str(i) for i in st.session_state.model._label_map.keys()]
                )

            info[t("selection.model_info.version")
                 ] = st.session_state.model.version
            st.json(info)
        else:
            st.warning(t("selection.no_model"))

    add_additional_info()


def model_selection_v2() -> None:
    """Simplified model selection with step-by-step filtering."""
    st.title(t("selection.v2.title"))

    # Initialize session state for filters
    if "v2_filters" not in st.session_state:
        st.session_state.v2_filters = {
            "dataset": None,
            "diffuser": None,
            "noise": None,
            "epochs": None,
            "model_file": None,
            "sampler": None
        }

    # Directory selection
    model_dir = st.text_input(
        t("selection.v2.model_dir"),
        value=st.session_state.get("model_dir", "examples/saved_models"),
        key="v2_dir"
    )
    st.session_state.model_dir = model_dir

    # Scan models and extract metadata
    @st.cache_data
    def scan_models(model_dir: str) -> List[Dict]:
        models = []
        dummy_model = GenerativeModel(verbose=False)

        try:
            for fname in os.listdir(model_dir):
                if not fname.endswith((".pt", ".pth")):
                    continue

                try:
                    path = os.path.join(model_dir, fname)
                    dummy_model._class_only_load(path)

                    # Extract components from filename
                    base = os.path.splitext(fname)[0]
                    parts = base.replace("-", "_").split("_")

                    model_info = {
                        "path": path,
                        "filename": fname,
                        "dataset": parts[0].lower() if len(parts) > 0 else "unknown",
                        "diffuser": _get_class_name(dummy_model.diffusion),
                        "requires_noise": dummy_model.diffusion.NEEDS_NOISE_SCHEDULE,
                        "noise": _get_class_name(dummy_model.diffusion.schedule)
                        if dummy_model.diffusion.NEEDS_NOISE_SCHEDULE
                        else None,
                        "epochs": parts[-1].split("e")[0],
                        "valid": True
                    }
                    models.append(model_info)
                except Exception as e:
                    models.append(
                        {"filename": fname, "valid": False, "error": str(e)})
        except FileNotFoundError:
            return []

        return models

    if st.button(f":material/search: {t('selection.v2.scan_models')}", key="scan_v2"):
        st.cache_data.clear()
        st.session_state.v2_filters = {
            k: None for k in st.session_state.v2_filters}

    models = scan_models(model_dir)

    print(f"Scanning directory: {model_dir}")
    print(os.listdir(model_dir))

    # Dataset selection
    previous_dataset = st.session_state.v2_filters["dataset"]
    datasets = sorted({m["dataset"] for m in models if m.get("valid")})
    dataset_display = [DATASET_NAMES.get(d, d.upper()) for d in datasets]
    selected_dataset = st.selectbox(
        t("selection.v2.dataset.label"),
        options=dataset_display,
        index=datasets.index(previous_dataset) if previous_dataset in datasets else (
            None if len(dataset_display) > 1 else 0),
        placeholder=t("selection.v2.dataset.placeholder"),
        disabled=len(dataset_display) == 1,
        key="v2_dataset"
    )

    # Handle dataset mapping safely
    if selected_dataset:
        if selected_dataset in DATASET_NAMES.values():
            matching_keys = [
                k for k, v in DATASET_NAMES.items() if v == selected_dataset]
            new_dataset = matching_keys[0] if matching_keys else selected_dataset.lower(
            )
        else:
            new_dataset = selected_dataset.lower()
    else:
        new_dataset = None

    if new_dataset != previous_dataset:
        st.session_state.v2_filters.update({
            "diffuser": None,
            "noise": None,
            "epochs": None,
            "model_file": None,
            "sampler": None
        })
    st.session_state.v2_filters["dataset"] = new_dataset

    if not st.session_state.v2_filters["dataset"]:
        return

    # Filter models based on current selections
    filtered = [m for m in models if m["valid"]]

    # Apply dataset filter
    if st.session_state.v2_filters["dataset"]:
        filtered = [m for m in filtered if m["dataset"]
                    == st.session_state.v2_filters["dataset"]]

    # Diffuser selection
    previous_diffuser = st.session_state.v2_filters["diffuser"]
    diffusers = sorted({m["diffuser"] for m in filtered})
    diffuser_display = [_get_formatted_name(
        d, DIFFUSER_NAMES) for d in diffusers]
    selected_diffuser = st.selectbox(
        t("selection.v2.diffuser.label"),
        options=diffuser_display,
        index=diffusers.index(previous_diffuser) if previous_diffuser in diffusers else (
            None if len(diffuser_display) > 1 else 0),
        placeholder=t("selection.v2.diffuser.placeholder"),
        disabled=(not st.session_state.v2_filters["dataset"]) or len(
            diffuser_display) == 1,
        key="v2_diffuser"
    )

    # Handle diffuser mapping
    if selected_diffuser:
        matching_diffusers = [
            k for k, v in DIFFUSER_NAMES.items() if v == selected_diffuser]
        new_diffuser = (
            matching_diffusers[0] if matching_diffusers
            else selected_diffuser.replace(" ", "").lower()
        )
    else:
        new_diffuser = None

    if new_diffuser != previous_diffuser:
        st.session_state.v2_filters.update({
            "noise": None,
            "epochs": None,
            "model_file": None,
            "sampler": None
        })
    st.session_state.v2_filters["diffuser"] = new_diffuser

    if not st.session_state.v2_filters["diffuser"]:
        return

    filtered = [m for m in filtered if m["diffuser"] == new_diffuser]

    # Noise schedule selection
    if filtered and filtered[0]["requires_noise"]:
        previous_noise = st.session_state.v2_filters["noise"]
        noises = sorted({m["noise"] for m in filtered if m["noise"]})
        noise_display = [_get_formatted_name(n, NOISE_NAMES) for n in noises]
        selected_noise = st.selectbox(
            t("selection.v2.noise.label"),
            options=noise_display,
            index=noises.index(previous_noise) if previous_noise in noises else (
                None if len(noise_display) > 1 else 0),
            placeholder=t("selection.v2.noise.placeholder"),
            disabled=(not st.session_state.v2_filters["diffuser"]) or len(
                noise_display) == 1,
            key="v2_noise"
        )

        if selected_noise:
            matching_noises = [
                k for k, v in NOISE_NAMES.items() if v == selected_noise]
            new_noise = (
                matching_noises[0] if matching_noises
                else selected_noise.replace(" ", "").lower()
            )
        else:
            new_noise = None

        if new_noise != previous_noise:
            st.session_state.v2_filters.update({
                "epochs": None,
                "model_file": None,
                "sampler": None
            })
        st.session_state.v2_filters["noise"] = new_noise

        if not st.session_state.v2_filters["noise"]:
            return

        filtered = [m for m in filtered if m["noise"] == new_noise]

    # Epochs selection
    if filtered:
        previous_epochs = st.session_state.v2_filters["epochs"]
        epochs = sorted({m["epochs"] for m in filtered if m["epochs"]},
                        key=lambda x: int(x) if x.isdigit() else 0)
        selected_epochs = st.selectbox(
            t("selection.v2.epochs.label"),
            options=epochs,
            index=epochs.index(previous_epochs) if previous_epochs in epochs else (
                None if len(epochs) > 1 else 0),
            placeholder=t("selection.v2.epochs.placeholder"),
            disabled=(not (st.session_state.v2_filters.get(
                "noise") or not filtered[0]["requires_noise"])) or len(epochs) == 1,
            key="v2_epochs"
        )

        if selected_epochs:
            filtered = [m for m in filtered if m["epochs"] == selected_epochs]
            new_epochs = selected_epochs
        else:
            new_epochs = None

        if new_epochs != previous_epochs:
            st.session_state.v2_filters.update({
                "model_file": None,
                "sampler": None
            })
        st.session_state.v2_filters["epochs"] = new_epochs

        if not st.session_state.v2_filters["epochs"]:
            return

    # Final model selection if multiple remain
    if filtered and len(filtered) > 1:
        previous_model_file = st.session_state.v2_filters["model_file"]
        model_names = [m["filename"] for m in filtered]
        selected_model = st.selectbox(
            t("selection.v2.model_select.label"),
            options=model_names,
            index=model_names.index(
                previous_model_file) if previous_model_file in model_names else None,
            placeholder=t("selection.v2.model_select.placeholder"),
            key="v2_model_select"
        )
        if selected_model:
            filtered = [m for m in filtered if m["filename"] == selected_model]
            new_model_file = selected_model
        else:
            new_model_file = None

        if new_model_file != previous_model_file:
            st.session_state.v2_filters["sampler"] = None
        st.session_state.v2_filters["model_file"] = new_model_file

        if not st.session_state.v2_filters["model_file"]:
            return

    # Sampler selection and load button
    if filtered and len(filtered) == 1:
        previous_sampler = st.session_state.v2_filters["sampler"]
        model = filtered[0]
        samplers = list(SAMPLER_NAMES.keys())
        samplers_display = list(SAMPLER_NAMES.values())
        selected_sampler = st.selectbox(
            t("selection.v2.sampler.label"),
            options=samplers_display,
            index=samplers.index(
                previous_sampler) if previous_sampler in samplers else None,
            placeholder=t("selection.v2.sampler.placeholder"),
            key="v2_sampler"
        )

        if selected_sampler:
            if selected_sampler in SAMPLER_NAMES.values():
                matching_keys = [
                    k for k, v in SAMPLER_NAMES.items() if v == selected_sampler]
                st.session_state.v2_filters["sampler"] = matching_keys[0] if matching_keys else selected_sampler.lower(
                )
            else:
                st.session_state.v2_filters["sampler"] = selected_sampler.lower(
                )

        if st.button(
            f":material/check: {t('selection.v2.load_button')}",
            disabled=not selected_sampler,
            help=t("selection.v2.load_help"),
            key="v2_load"
        ):
            try:
                model_inst = GenerativeModel(
                    diffusion=model["diffuser"],
                    noise_schedule=model["noise"] if model["requires_noise"] else None,
                    verbose=False
                )
                model_inst.load(model["path"], unsafe=True)

                model_inst.sampler = st.session_state.v2_filters["sampler"]

                st.session_state.model = model_inst
                st.session_state.model_name = os.path.splitext(model["filename"])[
                    0]
                st.success(t("selection.v2.load_success"))
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(t("selection.v2.invalid_model").format(str(e)))
    elif filtered:
        st.error(t("selection.v2.no_models"))


def generation() -> None:
    """
    Render the image generation page.

    Provides UI for generating images with the loaded model.
    """
    st.title(t("generation.title"))

    if st.session_state.model is None:
        st.warning(t("generation.no_model_warning"))
        return

    # Get label map from model data
    label_map = st.session_state.model._label_map

    with st.expander(
        t("generation.settings_expander"), expanded=True, icon=":material/tune:"
    ):
        col1, col2 = st.columns(2)

        with col1:
            num_images = st.slider(
                t("generation.num_images_slider"), 1, 16,
                st.session_state.settings["num_images"]
            )
            st.session_state.settings["num_images"] = num_images
            use_seed = st.checkbox(
                t("generation.use_seed_checkbox"), st.session_state.settings["use_seed"]
            )
            st.session_state.settings["use_seed"] = use_seed
            if use_seed:
                seed = st.number_input(
                    t("generation.seed_input"), st.session_state.settings["seed"]
                )
                st.session_state.settings["seed"] = seed
            else:
                seed = None
                st.number_input(
                    t("generation.seed_input"), st.session_state.settings["seed"],
                    disabled=True, key="generation_seed"
                )

        with col2:
            steps = st.slider(
                t("generation.steps_slider"), 10, 1000,
                st.session_state.settings["steps"]
            )
            st.session_state.settings["steps"] = steps
            show_progress = st.checkbox(
                t("generation.show_progress_checkbox"),
                st.session_state.settings["show_progress"]
            )
            st.session_state.settings["show_progress"] = show_progress

            # Class selection only if label map exists
            if label_map is not None:
                selected_class = st.selectbox(
                    t("generation.class_selectbox"),
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

    if st.button(t("generation.generate_button"), icon=":material/auto_awesome:"):
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

                    for idx, (img, col) in enumerate(zip(images_in_row, cols)):
                        with col:
                            st.image(img, use_container_width=True)

                            @st.fragment
                            def download_image(buf: io.BytesIO, n: int) -> None:
                                st.download_button(
                                    t("generation.download_button").format(n),
                                    buf.getvalue(),
                                    f"generated_{n}.png",
                                    "image/png",
                                    icon=":material/download:",
                                    key=f"dl_{n}",
                                    on_click=lambda: None
                                )

                            buf = io.BytesIO()
                            img.save(buf, format="PNG", compress_level=0)
                            download_image(buf, row_start+idx+1)

                    row_start += cols_in_row

        except Exception as e:
            st.error(t("generation.error_message").format(str(e)))


def colorization() -> None:
    """
    Render the image colorization page.

    Provides UI for colorizing grayscale images.
    """
    st.title(t("colorization.title"))

    if st.session_state.model is None:
        st.warning(t("colorization.no_model_warning"))
        return

    if st.session_state.model.num_channels != 3:
        st.warning(t("colorization.unsupported_model_warning"))
        return

    # Get label map from model data
    label_map = st.session_state.model._label_map

    # Settings at the top
    with st.expander(
        t("colorization.settings_expander"), expanded=True, icon=":material/tune:"
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div style='height: 100%;'></div>",
                        unsafe_allow_html=True)
            use_seed = st.checkbox(
                t("colorization.use_seed_checkbox"),
                st.session_state.settings["use_seed"]
            )
            st.session_state.settings["use_seed"] = use_seed
            if use_seed:
                seed = st.number_input(
                    t("colorization.seed_input"),
                    st.session_state.settings["seed"]
                )
                st.session_state.settings["seed"] = seed
            else:
                seed = None
                st.number_input(
                    t("colorization.seed_input"),
                    st.session_state.settings["seed"],
                    disabled=True,
                    key="colorization_seed"
                )

        with col2:
            steps = st.slider(
                t("colorization.steps_slider"), 10, 1000,
                st.session_state.settings["steps"]
            )
            st.session_state.settings["steps"] = steps
            show_progress = st.checkbox(
                t("colorization.show_progress_checkbox"),
                st.session_state.settings["show_progress"]
            )
            st.session_state.settings["show_progress"] = show_progress

            # Class selection only if label map exists
            if label_map is not None:
                selected_class = st.selectbox(
                    t("colorization.class_selectbox"),
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
        t("colorization.uploader_label"),
        type=["jpg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("L")

        with col1:
            st.image(
                image,
                caption=t("colorization.original_caption"),
                use_container_width=True
            )

            if st.button(t("colorization.colorize_button"), icon=":material/auto_awesome:"):
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
                            colored_img,
                            caption=t("colorization.result_caption"),
                            use_container_width=True
                        )

                        buf = io.BytesIO()
                        colored_img.save(buf, format="PNG")

                        @st.fragment
                        def create_download() -> None:
                            st.download_button(
                                t("colorization.download_button"),
                                buf.getvalue(),
                                "colorized.png",
                                "image/png",
                                icon=":material/download:",
                                key="colorized_dl"
                            )
                        create_download()

                except Exception as e:
                    st.error(t("colorization.error_message").format(str(e)))


def imputation() -> None:
    """
    Render the image imputation page.

    Provides UI for filling in transparent areas of images.
    """
    st.title(t("imputation.title"))

    if st.session_state.model is None:
        st.warning(t("imputation.no_model_warning"))
        return

    # Get label map from model data
    label_map = st.session_state.model._label_map

    # Settings at the top
    with st.expander(
        t("imputation.settings_expander"), expanded=True, icon=":material/tune:"
    ):
        col1, col2 = st.columns(2)

        with col1:
            use_seed = st.checkbox(
                t("imputation.use_seed_checkbox"),
                st.session_state.settings["use_seed"]
            )
            st.session_state.settings["use_seed"] = use_seed
            if use_seed:
                seed = st.number_input(
                    t("imputation.seed_input"),
                    st.session_state.settings["seed"]
                )
                st.session_state.settings["seed"] = seed
            else:
                seed = None
                st.number_input(
                    t("imputation.seed_input"),
                    st.session_state.settings["seed"],
                    disabled=True,
                    key="imputation_seed"
                )

        with col2:
            steps = st.slider(
                t("imputation.steps_slider"), 10, 1000,
                st.session_state.settings["steps"]
            )
            st.session_state.settings["steps"] = steps
            show_progress = st.checkbox(
                t("imputation.show_progress_checkbox"),
                st.session_state.settings["show_progress"]
            )
            st.session_state.settings["show_progress"] = show_progress

            # Class selection only if label map exists
            if label_map is not None:
                selected_class = st.selectbox(
                    t("imputation.class_selectbox"),
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
        t("imputation.uploader_label"),
        type=["png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGBA")
        width, height = image.size

        # Split into RGB and alpha channels
        rgb_img = image.convert("RGB")
        alpha_channel = np.array(image.split()[-1])

        # Create mask from transparency
        mask = (alpha_channel == 0)

        if not np.any(mask):
            st.warning(t("imputation.no_transparency_warning"))
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
                t("imputation.original_caption"),
                unsafe_allow_html=True
            )

            if st.button(t("imputation.impute_button"), icon=":material/auto_awesome:"):
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
                            imputed_img,
                            caption=t("imputation.result_caption"),
                            use_container_width=True
                        )

                        buf = io.BytesIO()
                        imputed_img.save(buf, format="PNG")

                        @st.fragment
                        def create_download() -> None:
                            st.download_button(
                                t("imputation.download_button"),
                                buf.getvalue(),
                                "imputed.png",
                                "image/png",
                                icon=":material/download:",
                                key="imputed_dl"
                            )
                        create_download()

                except Exception as e:
                    st.error(t("imputation.error_message").format(str(e)))


pages = {
    t("pages.management"): [
        st.Page(lambda: model_selection_v1() if st.session_state.get("mgmt_version", "v1").split(" ")[0] == "v1" else model_selection_v2(),
                title=t("pages.model_selection"), icon=":material/folder:"),
    ],
    t("pages.generation"): [
        st.Page(generation, title=t("pages.image_generation"),
                icon=":material/image:"),
        st.Page(colorization, title=t("pages.colorization"),
                icon=":material/palette:"),
        st.Page(imputation, title=t("pages.imputation"),
                icon=":material/draw:"),
    ]
}


def main() -> None:
    """
    Initialize and run the Streamlit dashboard.
    """
    st.set_page_config(
        page_title=t("main.page_title"),
        layout="wide",
        page_icon=":frame_with_picture:",
        initial_sidebar_state="expanded",
        menu_items={"about": "\n\n".join(ABOUT) + "\n\n---"}
    )

    primary_color = toml.load(
        ".streamlit/config.toml")["theme"]["primaryColor"]

    st.html(
        "<style>" + load_css() +
        "\na {\n    color: " + primary_color + " !important;\n}\n\n" +
        "code {\n    color: " + primary_color + " !important;\n}\n\n" +
        ":root {\n    --primary-color: " + primary_color + ";\n}</style>"
    )

    # Language selection dropdown in sidebar
    with st.sidebar:
        st.selectbox(
            t("main.language_selection"),
            AVAILABLE_LANGUAGES,
            key="language",
            format_func=lambda x: t(f"languages.{x}")
        )

        st.selectbox(
            t("main.management_version"),
            ["v1 - " + t("main.original"), "v2 - " + t("main.simplified")],
            key="mgmt_version",
            index=1
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
