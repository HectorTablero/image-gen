from image_gen.diffusion import (
    BaseDiffusion,
    VarianceExploding,
    VariancePreserving,
    SubVariancePreserving
)


def test_ve_implementation():
    ve = VarianceExploding()
    assert isinstance(ve, BaseDiffusion)
    assert ve.forward_process(None, None) is None
    assert ve.get_schedule(None) is None


def test_vp_implementation():
    vp = VariancePreserving()
    assert isinstance(vp, BaseDiffusion)
    assert vp.reverse_process(None, None) is None


def test_sub_vp_implementation():
    sub_vp = SubVariancePreserving()
    assert isinstance(sub_vp, BaseDiffusion)
    assert sub_vp.get_schedule(None) is None
