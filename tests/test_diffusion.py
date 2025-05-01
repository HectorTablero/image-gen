import pytest
import torch
import numpy as np
from .context import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_gen.diffusion import BaseDiffusion, VarianceExploding, VariancePreserving, SubVariancePreserving
from image_gen.noise import LinearNoiseSchedule, CosineNoiseSchedule

@pytest.fixture
def batch_size():
    return 5

@pytest.fixture
def shape():
    return (3, 16, 16)  # (C, H, W)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def linear_schedule():
    return LinearNoiseSchedule()

@pytest.fixture
def cosine_schedule():
    return CosineNoiseSchedule()

@pytest.fixture
def ve_diffusion():
    return VarianceExploding(sigma=25.0)

@pytest.fixture
def vp_diffusion(linear_schedule):
    return VariancePreserving(linear_schedule)

@pytest.fixture
def svp_diffusion(linear_schedule):
    return SubVariancePreserving(linear_schedule)

def test_ve_diffusion_initialization():
    """Test Variance Exploding diffusion initialization."""
    # Default initialization
    ve = VarianceExploding()
    assert hasattr(ve, "schedule")
    assert ve.schedule is not None
    
    # Custom initialization
    ve = VarianceExploding(sigma=10.0)
    assert ve.schedule.sigma == 10.0
    
    # Test that NEEDS_NOISE_SCHEDULE is False
    assert VarianceExploding.NEEDS_NOISE_SCHEDULE is False

def test_vp_diffusion_initialization(linear_schedule, cosine_schedule):
    """Test Variance Preserving diffusion initialization."""
    # Initialize with linear schedule
    vp = VariancePreserving(linear_schedule)
    assert vp.schedule == linear_schedule
    
    # Initialize with cosine schedule
    vp = VariancePreserving(cosine_schedule)
    assert vp.schedule == cosine_schedule
    
    # Test that NEEDS_NOISE_SCHEDULE is True
    assert VariancePreserving.NEEDS_NOISE_SCHEDULE is True

def test_svp_diffusion_initialization(linear_schedule, cosine_schedule):
    """Test Sub-Variance Preserving diffusion initialization."""
    # Initialize with linear schedule
    svp = SubVariancePreserving(linear_schedule)
    assert svp.schedule == linear_schedule
    
    # Initialize with cosine schedule
    svp = SubVariancePreserving(cosine_schedule)
    assert svp.schedule == cosine_schedule
    
    # Test that NEEDS_NOISE_SCHEDULE is True
    assert SubVariancePreserving.NEEDS_NOISE_SCHEDULE is True

def test_ve_forward_sde(ve_diffusion, batch_size, shape, device):
    """Test forward SDE for Variance Exploding diffusion."""
    x = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    drift, diffusion = ve_diffusion.forward_sde(x, t)
    
    # For VE, drift should be zero
    assert torch.allclose(drift, torch.zeros_like(drift))
    
    # Diffusion should be positive and shape should match input
    assert diffusion.shape == (batch_size, 1, 1, 1)
    assert (diffusion > 0).all()

def test_vp_forward_sde(vp_diffusion, batch_size, shape, device):
    """Test forward SDE for Variance Preserving diffusion."""
    x = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    drift, diffusion = vp_diffusion.forward_sde(x, t)
    
    # Drift should have same shape as input
    assert drift.shape == x.shape
    
    # Diffusion should be positive and shape should match
    assert diffusion.shape == (batch_size, 1, 1, 1)
    assert (diffusion > 0).all()

def test_svp_forward_sde(svp_diffusion, batch_size, shape, device):
    """Test forward SDE for Sub-Variance Preserving diffusion."""
    x = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    drift, diffusion = svp_diffusion.forward_sde(x, t)
    
    # Drift should have same shape as input
    assert drift.shape == x.shape
    
    # Diffusion should be positive and shape should match
    assert diffusion.shape == (batch_size, 1, 1, 1)
    assert (diffusion > 0).all()

def test_ve_forward_process(ve_diffusion, batch_size, shape, device):
    """Test forward process for Variance Exploding diffusion."""
    x0 = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    xt, noise = ve_diffusion.forward_process(x0, t)
    
    # Shapes should match
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    
    # Check if xt is not the same as x0 (noise was added)
    assert not torch.allclose(xt, x0)

def test_vp_forward_process(vp_diffusion, batch_size, shape, device):
    """Test forward process for Variance Preserving diffusion."""
    x0 = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    xt, noise = vp_diffusion.forward_process(x0, t)
    
    # Shapes should match
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    
    # Check if xt is not the same as x0 (it is a weighted combination)
    assert not torch.allclose(xt, x0)

def test_svp_forward_process(svp_diffusion, batch_size, shape, device):
    """Test forward process for Sub-Variance Preserving diffusion."""
    x0 = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    xt, noise = svp_diffusion.forward_process(x0, t)
    
    # Shapes should match
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape
    
    # Check if xt is not the same as x0 (it is a weighted combination)
    assert not torch.allclose(xt, x0)

def test_ve_compute_loss(ve_diffusion, batch_size, shape, device):
    """Test loss computation for Variance Exploding diffusion."""
    score = torch.randn(batch_size, *shape, device=device)
    noise = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    loss = ve_diffusion.compute_loss(score, noise, t)
    
    # Loss should be a 1D tensor with batch_size elements
    assert loss.shape == (batch_size,)
    assert torch.all(torch.isfinite(loss))

def test_vp_compute_loss(vp_diffusion, batch_size, shape, device):
    """Test loss computation for Variance Preserving diffusion."""
    score = torch.randn(batch_size, *shape, device=device)
    noise = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    loss = vp_diffusion.compute_loss(score, noise, t)
    
    # Loss should be a 1D tensor with batch_size elements
    assert loss.shape == (batch_size,)
    assert torch.all(torch.isfinite(loss))

def test_svp_compute_loss(svp_diffusion, batch_size, shape, device):
    """Test loss computation for Sub-Variance Preserving diffusion."""
    score = torch.randn(batch_size, *shape, device=device)
    noise = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    loss = svp_diffusion.compute_loss(score, noise, t)
    
    # Loss should be a 1D tensor with batch_size elements
    assert loss.shape == (batch_size,)
    assert torch.all(torch.isfinite(loss))

def test_backward_sde(ve_diffusion, vp_diffusion, svp_diffusion, batch_size, shape, device):
    """Test backward SDE for all diffusion types."""
    x = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    score = torch.randn(batch_size, *shape, device=device)
    
    # Test for each diffusion type
    for diffusion in [ve_diffusion, vp_diffusion, svp_diffusion]:
        drift, diffusion_coef = diffusion.backward_sde(x, t, score)
        
        # Drift and diffusion should have correct shapes
        assert drift.shape == x.shape
        assert diffusion_coef.shape == (batch_size, 1, 1, 1) or diffusion_coef.shape == x.shape
        
        # Values should be finite
        assert torch.all(torch.isfinite(drift))
        assert torch.all(torch.isfinite(diffusion_coef))

def test_end_to_end_process(ve_diffusion, batch_size, shape, device):
    """Test end-to-end forward and backward process."""
    x0 = torch.randn(batch_size, *shape, device=device)
    t = torch.rand(batch_size, device=device)
    
    # Forward process
    xt, noise = ve_diffusion.forward_process(x0, t)
    
    # Create mock score function (ideal score would be -noise)
    score = -noise
    
    # Backward SDE step
    drift, diffusion_coef = ve_diffusion.backward_sde(xt, t, score)
    
    # Check that drift pushes in the right direction
    delta_t = 0.01
    x_prev = xt + drift * (-delta_t)
    
    # x_prev should be closer to x0 than xt is
    assert torch.norm(x_prev - x0) < torch.norm(xt - x0)