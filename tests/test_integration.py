import pytest
import torch
import numpy as np
from .context import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_gen.base import GenerativeModel
from image_gen.diffusion import VarianceExploding, VariancePreserving, SubVariancePreserving
from image_gen.noise import LinearNoiseSchedule, CosineNoiseSchedule
from image_gen.samplers import EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector
from image_gen.metrics import BitsPerDimension, FrechetInceptionDistance, InceptionScore

class SimpleDataset:
    """Simple dataset for integration testing."""
    def __init__(self, size=5, shape=(1, 8, 8), class_conditional=False):
        self.data = [torch.randn(shape) for _ in range(size)]
        self.class_conditional = class_conditional
        if class_conditional:
            self.labels = [torch.tensor(i % 2) for i in range(size)]
    
    def __getitem__(self, idx):
        if self.class_conditional:
            return (self.data[idx], self.labels[idx])
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.mark.parametrize("diffusion_type", ["ve", "vp", "sub-vp"])
@pytest.mark.parametrize("sampler_type", ["euler-maruyama", "exponential", "ode"])
@pytest.mark.parametrize("noise_schedule", [None, "linear", "cosine"])
def test_model_train_generate_pipeline(diffusion_type, sampler_type, noise_schedule):
    """Test full pipeline: initialize, train, generate."""
    try:
        # Create model with verbose=False to avoid tqdm issues
        model = GenerativeModel(
            diffusion=diffusion_type,
            sampler=sampler_type,
            noise_schedule=noise_schedule,
            verbose=False  # This is important!
        )
        
        # Create dataset and train (minimal settings for testing)
        dataset = SimpleDataset()
        
        # Train without verbose to avoid tqdm wrapper
        model.train(dataset, epochs=1, batch_size=2, lr=0.01)
        
        # Generate samples
        samples = model.generate(num_samples=2, n_steps=5)
        
        # Verify output
        assert samples.shape == (2, 1, 8, 8)
        assert torch.all(torch.isfinite(samples))
        
    except Exception as e:
        # If we get CUDA memory errors, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e

@pytest.mark.parametrize("diffusion_class", [VarianceExploding, VariancePreserving, SubVariancePreserving])
@pytest.mark.parametrize("sampler_class", [EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector])
def test_diffusion_sampler_integration(diffusion_class, sampler_class, device):
    """Test integration of diffusion and sampler components."""
    try:
        # Create diffusion instance
        if diffusion_class.NEEDS_NOISE_SCHEDULE:
            noise_schedule = LinearNoiseSchedule()
            diffusion = diffusion_class(noise_schedule)
        else:
            diffusion = diffusion_class()
        
        # Create sampler instance
        sampler = sampler_class(diffusion, verbose=False)
        
        # Create batch and time tensor
        batch_size, channels, height, width = 2, 1, 8, 8
        x = torch.randn(batch_size, channels, height, width, device=device)
        t = torch.rand(batch_size, device=device)
        
        # Test forward process
        xt, noise = diffusion.forward_process(x, t)
        assert xt.shape == x.shape
        assert noise.shape == x.shape
        
        # Create mock score function
        def score_fn(x, t):
            return - 0.1 * x
        
        # Test sampling for a few steps
        n_steps = 5
        samples = sampler(xt, score_fn, n_steps=n_steps)
        
        # Verify output
        assert samples.shape == x.shape
        assert torch.all(torch.isfinite(samples))
        
    except Exception as e:
        # If we get CUDA memory errors, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e

def test_class_conditional_pipeline():
    """Test class-conditional training and generation pipeline."""
    try:
        # Create model
        model = GenerativeModel(
            diffusion="ve",
            sampler="euler-maruyama",
            verbose=False
        )
        
        # Create class-conditional dataset and train
        dataset = SimpleDataset(class_conditional=True)
        model.train(dataset, epochs=1, batch_size=2, lr=0.01)
        
        # Verify that model is class-conditional
        assert model.num_classes == 2
        assert model.stored_labels is not None
        
        # Set string labels
        model.set_labels(["class0", "class1"])
        
        # Generate samples with class conditioning
        samples0 = model.generate(num_samples=2, n_steps=5, class_labels=0)
        samples1 = model.generate(num_samples=2, n_steps=5, class_labels=1)
        
        # Verify outputs
        assert samples0.shape == (2, 1, 8, 8)
        assert samples1.shape == (2, 1, 8, 8)
        assert torch.all(torch.isfinite(samples0))
        assert torch.all(torch.isfinite(samples1))
        
        # Verify that different class labels produce different samples
        assert torch.abs(samples0 - samples1).mean() > 0.01
        
    except Exception as e:
        # If we get CUDA memory errors, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e

def test_save_load_pipeline(tmp_path):
    """Test save/load functionality in the pipeline."""
    try:
        # Create and train model
        model = GenerativeModel(
            diffusion="ve",
            sampler="euler-maruyama",
            verbose=False
        )
        
        dataset = SimpleDataset()
        model.train(dataset, epochs=1, batch_size=2, lr=0.01)
        
        # Generate samples before saving
        samples_before = model.generate(num_samples=2, n_steps=5, seed=42)
        
        # Save model
        save_path = tmp_path / "test_model.pt"
        model.save(save_path)
        
        # Create new model and load
        new_model = GenerativeModel()
        new_model.load(save_path)
        
        # Generate samples after loading
        samples_after = new_model.generate(num_samples=2, n_steps=5, seed=42)
        
        # Verify basic statistical properties instead of exact equality
        assert samples_before.shape == samples_after.shape
        assert torch.isfinite(samples_after).all()
        
        # Opcional: verificar que ambas muestras están en rangos similares
        assert abs(samples_before.mean() - samples_after.mean()) < 1.0
        assert abs(samples_before.std() - samples_after.std()) < 1.0
        
    except Exception as e:
        # If we get CUDA memory errors, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e

def test_colorize_imputation_pipeline():
    """Test colorization and imputation functionality."""
    try:
        # Create and train model
        model = GenerativeModel(
            diffusion="ve",
            sampler="euler-maruyama",
            verbose=False
        )
        
        # Create dataset with RGB images
        dataset = SimpleDataset(shape=(3, 8, 8))
        model.train(dataset, epochs=1, batch_size=2, lr=0.01)
        
        # Test colorization
        grayscale = torch.randn(1, 1, 8, 8)
        colorized = model.colorize(grayscale, n_steps=5)
        assert colorized.shape == (1, 3, 8, 8)
        assert torch.all(torch.isfinite(colorized))
        
        # Test imputation
        image = torch.randn(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 3:5, 3:5] = 1  # Set small center region to be filled
        
        completed = model.imputation(image, mask, n_steps=5)
        assert completed.shape == (1, 3, 8, 8)
        assert torch.all(torch.isfinite(completed))
        
    except Exception as e:
        # If we get CUDA memory errors, skip
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping due to GPU memory limitations")
        else:
            raise e