import pytest
import torch
import numpy as np
from .context import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_gen.samplers import BaseSampler, EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector
from image_gen.diffusion import VarianceExploding, VariancePreserving, SubVariancePreserving
from image_gen.noise import LinearNoiseSchedule

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def shape():
    return (3, 16, 16)  # (C, H, W)

@pytest.fixture
def ve_diffusion():
    return VarianceExploding(sigma=25.0)

@pytest.fixture
def vp_diffusion():
    linear_schedule = LinearNoiseSchedule()
    return VariancePreserving(linear_schedule)

@pytest.fixture
def svp_diffusion():
    linear_schedule = LinearNoiseSchedule()
    return SubVariancePreserving(linear_schedule)

@pytest.fixture
def euler_sampler(ve_diffusion):
    return EulerMaruyama(ve_diffusion)

@pytest.fixture
def exponential_sampler(ve_diffusion):
    return ExponentialIntegrator(ve_diffusion)

@pytest.fixture
def ode_sampler(ve_diffusion):
    return ODEProbabilityFlow(ve_diffusion)

@pytest.fixture
def predictor_corrector_sampler(ve_diffusion):
    return PredictorCorrector(ve_diffusion)

@pytest.fixture
def mock_score_model(device):
    """Create a mock score model function that returns -x."""
    def score_fn(x, t):
        return -x
    return score_fn

# In tests/test_samplers.py
def test_sampler_initialization(ve_diffusion, vp_diffusion, svp_diffusion):
    """Test initialization of samplers with different diffusion processes."""
    for diffusion in [ve_diffusion, vp_diffusion, svp_diffusion]:
        for sampler_class in [EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector]:
            # Test default initialization (should have verbose=True)
            sampler = sampler_class(diffusion)
            assert sampler.diffusion == diffusion
            
            # Test with verbose=False explicitly
            sampler = sampler_class(diffusion, verbose=False)
            assert sampler.verbose is False

def test_euler_maruyama_sampling(euler_sampler, mock_score_model, batch_size, shape, device):
    """Test EulerMaruyama sampling process."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Sample with minimal steps for testing
    n_steps = 10
    samples = euler_sampler(x_T, mock_score_model, n_steps=n_steps)
    
    # Verify shape of output
    assert samples.shape == x_T.shape
    
    # Verify output is finite
    assert torch.all(torch.isfinite(samples))
    
    # Verify sampling with seed reproducibility
    samples1 = euler_sampler(x_T, mock_score_model, n_steps=n_steps, seed=42)
    samples2 = euler_sampler(x_T, mock_score_model, n_steps=n_steps, seed=42)
    assert torch.allclose(samples1, samples2)
    
    # Verify different seeds produce different outputs
    samples3 = euler_sampler(x_T, mock_score_model, n_steps=n_steps, seed=43)
    assert not torch.allclose(samples1, samples3)

def test_exponential_integrator_sampling(exponential_sampler, mock_score_model, batch_size, shape, device):
    """Test ExponentialIntegrator sampling process."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Sample with minimal steps for testing
    n_steps = 10
    samples = exponential_sampler(x_T, mock_score_model, n_steps=n_steps)
    
    # Verify shape of output
    assert samples.shape == x_T.shape
    
    # Verify output is finite
    assert torch.all(torch.isfinite(samples))
    
    # Verify sampling with seed reproducibility
    samples1 = exponential_sampler(x_T, mock_score_model, n_steps=n_steps, seed=42)
    samples2 = exponential_sampler(x_T, mock_score_model, n_steps=n_steps, seed=42)
    assert torch.allclose(samples1, samples2)

def test_ode_sampling(ode_sampler, mock_score_model, batch_size, shape, device):
    """Test ODEProbabilityFlow sampling process."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Sample with minimal steps for testing
    n_steps = 10
    samples = ode_sampler(x_T, mock_score_model, n_steps=n_steps)
    
    # Verify shape of output
    assert samples.shape == x_T.shape
    
    # Verify output is finite
    assert torch.all(torch.isfinite(samples))
    
    # For ODE, output should be deterministic even without seed
    samples1 = ode_sampler(x_T, mock_score_model, n_steps=n_steps)
    samples2 = ode_sampler(x_T, mock_score_model, n_steps=n_steps)
    assert torch.allclose(samples1, samples2, atol=1e-6)

def test_predictor_corrector_sampling(predictor_corrector_sampler, mock_score_model, batch_size, shape, device):
    """Test PredictorCorrector sampling process."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Sample with minimal steps for testing
    n_steps = 10
    samples = predictor_corrector_sampler(x_T, mock_score_model, n_steps=n_steps)
    
    # Verify shape of output
    assert samples.shape == x_T.shape
    
    # Verify output is finite
    assert torch.all(torch.isfinite(samples))
    
    # Verify sampling with seed reproducibility
    samples1 = predictor_corrector_sampler(x_T, mock_score_model, n_steps=n_steps, seed=42)
    samples2 = predictor_corrector_sampler(x_T, mock_score_model, n_steps=n_steps, seed=42)
    assert torch.allclose(samples1, samples2)

def test_callback_functionality(euler_sampler, mock_score_model, batch_size, shape, device):
    """Test callback functionality during sampling."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Create a simple callback that saves intermediate steps
    callback_steps = []
    def callback(x, step):
        callback_steps.append(step)
    
    # Sample with callback
    n_steps = 10
    callback_frequency = 2
    samples = euler_sampler(
        x_T, 
        mock_score_model, 
        n_steps=n_steps, 
        callback=callback, 
        callback_frequency=callback_frequency
    )
    
    # Verify callback was called at expected steps
    expected_steps = list(range(0, n_steps, callback_frequency))
    assert callback_steps == expected_steps

def test_guidance_functionality(euler_sampler, mock_score_model, batch_size, shape, device):
    """Test guidance functionality during sampling."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Create a simple guidance function that adds a constant
    constant = 0.1
    def guidance(x, t):
        return x + constant
    
    # Sample with guidance
    n_steps = 10
    samples_with_guidance = euler_sampler(
        x_T, 
        mock_score_model, 
        n_steps=n_steps, 
        guidance=guidance
    )
    
    # Sample without guidance
    samples_without_guidance = euler_sampler(
        x_T, 
        mock_score_model, 
        n_steps=n_steps
    )
    
    # Verify samples are different when guidance is applied
    assert not torch.allclose(samples_with_guidance, samples_without_guidance)

def test_samplers_with_different_diffusions(mock_score_model, batch_size, shape, device, ve_diffusion, vp_diffusion, svp_diffusion):
    """Test all samplers with different diffusion processes."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Minimal steps for testing
    n_steps = 5
    
    # Test all combinations of samplers and diffusions
    for diffusion in [ve_diffusion, vp_diffusion, svp_diffusion]:
        for sampler_class in [EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector]:
            sampler = sampler_class(diffusion)
            
            # Sample
            samples = sampler(x_T, mock_score_model, n_steps=n_steps)
            
            # Verify output
            assert samples.shape == x_T.shape
            assert torch.all(torch.isfinite(samples))

# In tests/test_samplers.py
def test_handle_nan_scores(euler_sampler, batch_size, shape, device):
    """Test that samplers can handle NaN scores."""
    # Create random noise as starting point
    x_T = torch.randn(batch_size, *shape, device=device)
    
    # Track if NaNs were encountered
    nan_encountered = [False]
    
    # Create a score model that returns NaNs for specific inputs
    def bad_score_model(x, t):
        # Create a detached copy to avoid requires_grad issues
        score = -x.detach().clone()
        
        # Inject NaNs in a deterministic way
        if t[0].item() > 0.5:  # Only for certain timesteps
            score[0, 0, 0, 0] = float('nan')
            nan_encountered[0] = True
            
        return score
    
    # Sample with minimal steps
    n_steps = 10  # More steps to ensure we hit NaN values
    samples = euler_sampler(x_T, bad_score_model, n_steps=n_steps)
    
    # Verify we encountered NaNs but got finite output
    assert nan_encountered[0], "NaNs were not encountered in the test"
    assert torch.all(torch.isfinite(samples)).item(), "Samples contain NaN values"