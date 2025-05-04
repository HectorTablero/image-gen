import pytest
import torch
import numpy as np
from . import context  # Add the image_gen module to the path

from image_gen.metrics import BaseMetric, BitsPerDimension, FrechetInceptionDistance, InceptionScore
from image_gen.base import GenerativeModel
from image_gen.diffusion import VarianceExploding


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
    m = GenerativeModel()
    # Create dummy dataset and train for 1 epoch
    batch_size = 2
    shape = (3, 32, 32)
    dataset = [(torch.rand(shape), torch.tensor(0)) for _ in range(4)]
    m.train(dataset, epochs=1, batch_size=batch_size)
    return m


def test_bpd_initialization(model):
    """Test BitsPerDimension initialization requires a model."""
    # must pass model
    bpd = BitsPerDimension(model=model)
    assert isinstance(bpd, BitsPerDimension)

    # passing None or omitting model should error
    with pytest.raises(TypeError):
        BitsPerDimension()  # missing required positional arg


def test_bpd_call(model, real_images, generated_images):
    """Test BitsPerDimension calculation (ignores generated)."""
    bpd = BitsPerDimension(model=model)
    # call signature now __call__(real, generated)
    score = bpd(real_images, generated_images)
    assert isinstance(score, float)
    assert score > 0


def test_bpd_dataset_input(model):
    """Test BitsPerDimension on dataset-like input."""
    # create small dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, idx): return torch.rand(3, 32, 32), 0
    ds = DummyDataset()
    bpd = BitsPerDimension(model=model)
    score = bpd(ds, None)
    assert isinstance(score, float)


def test_fid_initialization(model):
    """Test FrechetInceptionDistance initialization."""
    fid = FrechetInceptionDistance(model=model)
    assert fid.dims == 2048

    fid2 = FrechetInceptionDistance(model=model, dims=1024)
    assert fid2.dims == 1024


def test_fid_call(model, real_images, generated_images):
    """Test FrechetInceptionDistance calculation path."""
    fid = FrechetInceptionDistance(model=model, dims=128)
    # override activations to avoid heavy Inception
    real_acts = np.random.randn(real_images.shape[0], fid.dims)
    gen_acts = np.random.randn(generated_images.shape[0], fid.dims)
    # test private calculation
    score = fid._calculate_fid(real_acts, gen_acts)
    assert isinstance(score, float)

    # test full call with tensor inputs
    try:
        score_full = fid(real_images, generated_images)
        assert isinstance(score_full, float)
    except Exception as e:
        if "downloading" in str(e).lower() or "out of memory" in str(e).lower():
            pytest.skip("Skipping FID full call due to resource limitations")
        else:
            raise


def test_inception_score_initialization(model):
    """Test InceptionScore initialization."""
    is_metric = InceptionScore(model=model)
    assert is_metric.n_splits == 10

    is2 = InceptionScore(model=model, n_splits=5)
    assert is2.n_splits == 5


def test_inception_score_call(model, generated_images):
    """Test InceptionScore calculation (mean only)."""
    is_metric = InceptionScore(model=model, n_splits=4)
    # create dummy predictions to test private method
    preds = np.random.rand(generated_images.shape[0], 1000)
    preds = preds / preds.sum(axis=1, keepdims=True)
    mean, std = is_metric._calculate_is(preds)
    assert isinstance(mean, float)
    assert isinstance(std, float)

    # test full call via __call__(None, generated)
    try:
        score = is_metric(None, generated_images)
        assert isinstance(score, float)
    except Exception as e:
        if "downloading" in str(e).lower() or "out of memory" in str(e).lower():
            pytest.skip("Skipping IS full call due to resource limitations")
        else:
            raise


def test_metrics_names(model):
    """Test name and is_lower_better properties."""
    bpd = BitsPerDimension(model=model)
    fid = FrechetInceptionDistance(model=model)
    is_metric = InceptionScore(model=model)

    assert bpd.name == "Bits Per Dimension"
    assert bpd.is_lower_better is True

    assert fid.name == "Fréchet Inception Distance"
    assert fid.is_lower_better is True

    assert is_metric.name == "Inception Score"
    assert is_metric.is_lower_better is False
