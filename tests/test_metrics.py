import pytest
import torch
import numpy as np
from . import context  # Add the image_gen module to the path

from image_gen.metrics import BaseMetric, BitsPerDimension, FrechetInceptionDistance, InceptionScore
from image_gen.base import GenerativeModel
from image_gen.diffusion import VarianceExploding


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def real_images():
    """Generate a batch of fake 'real' images."""
    batch_size = 5
    shape = (3, 32, 32)
    return torch.rand(batch_size, *shape)


@pytest.fixture
def generated_images():
    """Generate a batch of fake 'generated' images."""
    batch_size = 5
    shape = (3, 32, 32)
    return torch.rand(batch_size, *shape)


@pytest.fixture
def model():
    """Create a minimal trained model for BPD testing."""
    model = GenerativeModel()

    # Create dummy dataset and train for 1 epoch
    batch_size = 2
    shape = (3, 32, 32)
    dataset = [(torch.rand(shape), torch.tensor(0)) for _ in range(4)]
    model.train(dataset, epochs=1, batch_size=batch_size)

    return model


def test_bpd_initialization(model):
    """Test BitsPerDimension initialization."""
    # Test with model
    bpd = BitsPerDimension(model=model)

    # Test with diffusion model
    bpd = BitsPerDimension(model=model, diffusion_model=model.diffusion)

    # Test with no model (should not raise error yet)
    bpd = BitsPerDimension()


def test_bpd_call(model, real_images):
    """Test BitsPerDimension calculation."""
    # Initialize with both model and diffusion_model
    bpd = BitsPerDimension(model=model, diffusion_model=model.diffusion)

    # Calculate BPD
    try:
        score = bpd(real_images)
        assert isinstance(score, float)
        assert score > 0  # BPD should be positive
    except Exception as e:
        # If test fails with GPU memory error, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e

    # Calculate BPD
    try:
        score = bpd(real_images)
        assert isinstance(score, float)
        assert score > 0  # BPD should be positive
    except Exception as e:
        # If test fails with GPU memory error, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e


def test_bpd_error_without_model():
    """Test BitsPerDimension error when no model is provided."""
    bpd = BitsPerDimension()

    with pytest.raises(ValueError):
        bpd(torch.rand(2, 3, 32, 32))


def test_fid_initialization():
    """Test FrechetInceptionDistance initialization."""
    # Test with default parameters
    fid = FrechetInceptionDistance()
    assert fid.device in ['cuda', 'cpu']
    assert fid.dims == 2048

    # Test with custom parameters
    fid = FrechetInceptionDistance(device='cpu', dims=1024)
    assert fid.device == 'cpu'
    assert fid.dims == 1024


def test_fid_call(real_images, generated_images):
    """Test FrechetInceptionDistance calculation."""
    fid = FrechetInceptionDistance(device='cpu')

    try:
        # Skip model initialization to avoid downloading Inception
        # Just test the calculation logic
        real_activations = np.random.randn(real_images.shape[0], 2048)
        gen_activations = np.random.randn(generated_images.shape[0], 2048)

        score = fid._calculate_fid(real_activations, gen_activations)
        assert isinstance(score, float)
    except Exception as e:
        if "CUDA out of memory" in str(e) or "downloading" in str(e).lower():
            pytest.skip("Skipping due to network or memory limitations")
        else:
            raise e


def test_inception_score_initialization():
    """Test InceptionScore initialization."""
    # Test with default parameters
    is_metric = InceptionScore()
    assert is_metric.device in ['cuda', 'cpu']
    assert is_metric.n_splits == 10

    # Test with custom parameters
    is_metric = InceptionScore(device='cpu', n_splits=5)
    assert is_metric.device == 'cpu'
    assert is_metric.n_splits == 5


def test_inception_score_call(generated_images):
    """Test InceptionScore calculation."""
    is_metric = InceptionScore(device='cpu')

    try:
        # Skip model initialization to avoid downloading Inception
        # Just test the calculation logic
        predictions = np.random.rand(generated_images.shape[0], 1000)
        # Normalize to make it a probability distribution
        predictions = predictions / predictions.sum(axis=1, keepdims=True)

        score_mean, score_std = is_metric._calculate_is(predictions)
        assert isinstance(score_mean, float)
        assert isinstance(score_std, float)
        assert score_mean > 0
    except Exception as e:
        if "CUDA out of memory" in str(e) or "downloading" in str(e).lower():
            pytest.skip("Skipping due to network or memory limitations")
        else:
            raise e


def test_metrics_names():
    """Test name property of metrics."""
    assert BitsPerDimension().name == "BitsPerDimension"
    assert FrechetInceptionDistance().name == "FrechetInceptionDistance"
    assert InceptionScore().name == "InceptionScore"
