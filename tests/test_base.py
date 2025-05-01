import pytest
import torch
import numpy as np
from .context import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_gen.base import GenerativeModel
from image_gen.diffusion import VarianceExploding, VariancePreserving, SubVariancePreserving
from image_gen.noise import LinearNoiseSchedule, CosineNoiseSchedule
from image_gen.samplers import EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector

class DummyDataset:
    """Dummy dataset for testing model training."""
    def __init__(self, shape=(3, 32, 32), size=10, class_conditional=False):
        self.shape = shape
        self.size = size
        self.class_conditional = class_conditional
        self.data = [torch.randn(shape) for _ in range(size)]
        if class_conditional:
            self.labels = [torch.tensor(i % 3) for i in range(size)]
    
    def __getitem__(self, idx):
        if self.class_conditional:
            return (self.data[idx], self.labels[idx])
        return self.data[idx]
    
    def __len__(self):
        return self.size

@pytest.fixture
def model():
    """Basic generative model for testing."""
    return GenerativeModel(diffusion="ve", sampler="euler-maruyama")

@pytest.fixture
def trained_model():
    """Pre-trained model for testing generation."""
    model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")
    dataset = DummyDataset(shape=(3, 32, 32), size=5)
    model.train(dataset, epochs=1, batch_size=2)
    return model

@pytest.fixture
def class_conditional_model():
    """Pre-trained class-conditional model for testing."""
    model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")
    dataset = DummyDataset(shape=(3, 32, 32), size=5, class_conditional=True)
    model.train(dataset, epochs=1, batch_size=2)
    model.set_labels(["cat", "dog", "bird"])
    return model

def test_model_initialization():
    """Test initialization with different parameters."""
    # Test default initialization
    model = GenerativeModel()
    assert model.diffusion.__class__.__name__ == "VarianceExploding"
    assert model.sampler.__class__.__name__ == "EulerMaruyama"
    
    # Test initialization with string parameters
    model = GenerativeModel(diffusion="vp", sampler="ode")
    assert model.diffusion.__class__.__name__ == "VariancePreserving"
    assert model.sampler.__class__.__name__ == "ODEProbabilityFlow"
    
    # Test initialization with class types
    model = GenerativeModel(diffusion=SubVariancePreserving, sampler=PredictorCorrector)
    assert model.diffusion.__class__.__name__ == "SubVariancePreserving"
    assert model.sampler.__class__.__name__ == "PredictorCorrector"

def test_noise_schedule_initialization():
    """Test initialization with different noise schedules."""
    # Test with linear noise schedule
    model = GenerativeModel(diffusion="vp", noise_schedule="linear")
    assert model.diffusion.schedule.__class__.__name__ == "LinearNoiseSchedule"
    
    # Test with cosine noise schedule
    model = GenerativeModel(diffusion="vp", noise_schedule="cosine")
    assert model.diffusion.schedule.__class__.__name__ == "CosineNoiseSchedule"
    
    # Test with class type
    model = GenerativeModel(diffusion="vp", noise_schedule=CosineNoiseSchedule)
    assert model.diffusion.schedule.__class__.__name__ == "CosineNoiseSchedule"

def test_model_properties(model):
    """Test model properties."""
    assert model.device is not None
    assert model.version is not None
    assert model.verbose is True
    
    # Change verbose setting
    model.verbose = False
    assert model.verbose is False
    assert model.sampler.verbose is False

def test_model_training(model):
    """Test model training process."""
    dataset = DummyDataset(shape=(3, 32, 32), size=5)
    model.train(dataset, epochs=1, batch_size=2)
    
    # Verify that the model was built correctly
    assert model.model is not None
    assert model.num_channels == 3
    assert model.shape == (32, 32)
    assert model.num_classes is None  # No class conditioning

def test_class_conditional_training():
    """Test class-conditional model training."""
    model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")
    dataset = DummyDataset(shape=(3, 32, 32), size=5, class_conditional=True)
    model.train(dataset, epochs=1, batch_size=2)
    
    # Verify that class conditioning works
    assert model.num_classes == 3  # Based on our dummy dataset
    assert model.stored_labels is not None
    
    # Test setting string labels
    model.set_labels(["cat", "dog", "bird"])
    assert len(model.labels) == 3

def test_model_generation(trained_model):
    """Test image generation."""
    samples = trained_model.generate(num_samples=2, n_steps=10)
    
    # Verify output shape and range
    assert samples.shape == (2, 3, 32, 32)
    assert not torch.isnan(samples).any()
    
    # Test with class conditioning
    class_model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")
    class_dataset = DummyDataset(shape=(3, 32, 32), size=5, class_conditional=True)
    class_model.train(class_dataset, epochs=1, batch_size=2)
    
    class_samples = class_model.generate(num_samples=2, n_steps=10, class_labels=0)
    assert class_samples.shape == (2, 3, 32, 32)
    assert not torch.isnan(class_samples).any()

def test_save_load(trained_model, tmp_path):
    """Test saving and loading model."""
    save_path = tmp_path / "test_model.pt"
    trained_model.save(save_path)
    
    # Create a new model and load
    new_model = GenerativeModel()
    new_model.load(save_path)
    
    # Verify that parameters were loaded correctly
    assert new_model.diffusion.__class__.__name__ == trained_model.diffusion.__class__.__name__
    assert new_model.num_channels == trained_model.num_channels
    assert new_model.shape == trained_model.shape
    
    # Test generation with loaded model
    samples = new_model.generate(num_samples=1, n_steps=5)
    assert samples.shape == (1, 3, 32, 32)
    assert not torch.isnan(samples).any()

def test_image_colorization(trained_model):
    """Test image colorization functionality."""
    # Create grayscale input
    grayscale = torch.randn(1, 1, 32, 32)
    
    # Test colorization
    colorized = trained_model.colorize(grayscale, n_steps=10)
    
    # Verify output shape and values
    assert colorized.shape == (1, 3, 32, 32)
    assert not torch.isnan(colorized).any()

def test_image_imputation(trained_model):
    """Test image imputation (inpainting) functionality."""
    # Create input image and mask
    image = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:20, 10:20] = 1  # Set center region to be filled
    
    # Test imputation
    completed = trained_model.imputation(image, mask, n_steps=10)
    
    # Verify output shape and values
    assert completed.shape == (1, 3, 32, 32)
    assert not torch.isnan(completed).any()

def test_diffusion_switching():
    """Test switching diffusion after initialization."""
    model = GenerativeModel(diffusion="ve")
    
    # Store the original diffusion type
    original_diffusion = model.diffusion.__class__.__name__
    
    # Build the model
    model._build_default_model(shape=(3, 32, 32))
    
    # Try to change diffusion (should be ignored after model is built)
    model.diffusion = "vp"
    
    # Verify diffusion type hasn't changed
    assert model.diffusion.__class__.__name__ == original_diffusion

def test_sampler_switching(trained_model):
    """Test switching sampler after training."""
    original_sampler = trained_model.sampler.__class__.__name__
    
    # Test changing sampler
    trained_model.sampler = "ode"
    assert trained_model.sampler.__class__.__name__ == "ODEProbabilityFlow"
    
    # Verify model still works
    samples = trained_model.generate(num_samples=1, n_steps=5)
    assert samples.shape == (1, 3, 32, 32)
    assert not torch.isnan(samples).any()

def test_guidance_scale(class_conditional_model):
    """Test different guidance scales for class-conditional generation."""
    # Generate with different guidance scales
    samples1 = class_conditional_model.generate(
        num_samples=1, n_steps=5, class_labels=0, guidance_scale=1.0
    )
    samples2 = class_conditional_model.generate(
        num_samples=1, n_steps=5, class_labels=0, guidance_scale=5.0
    )
    
    # Both should produce valid images
    assert not torch.isnan(samples1).any()
    assert not torch.isnan(samples2).any()
    
    # But they should be different (guidance affects output)
    assert torch.abs(samples1 - samples2).mean() > 0.01

def test_loss_function(trained_model):
    """Test loss function calculation."""
    # Create dummy batch
    batch = torch.randn(2, 3, 32, 32).to(trained_model.device)
    
    # Calculate loss
    loss = trained_model.loss_function(batch)
    
    # Verify loss is a scalar tensor with finite value
    assert loss.dim() == 0
    assert torch.isfinite(loss)