import pytest
import torch
from lit_models import LitFullModel

@pytest.fixture
def lit_model():
    return LitFullModel()

def test_shared_eval_for_checking(lit_model):
    # Create a sample batch
    batch = (
        (torch.randn(128, 10), torch.randn(128, 5), torch.arange(128), torch.tensor([0])),
        (torch.randn(128), torch.randn(128), torch.randn(128), torch.randn(128), torch.randn(128))
    )
    batch_idx = 0

    # Call the method
    lit_model.shared_eval_for_checking(batch, batch_idx)

    # Assert that the log has been updated correctly
    assert 'output' in lit_model.logged_results
    assert 'label' in lit_model.logged_results
    assert 'survival_time' in lit_model.logged_results
    assert 'vital_status' in lit_model.logged_results
    assert 'project_id' in lit_model.logged_results
    assert lit_model.logged_results['output'].shape == torch.Size([128])
    assert lit_model.logged_results['label'].shape == torch.Size([128])
    assert lit_model.logged_results['survival_time'].shape == torch.Size([128])
    assert lit_model.logged_results['vital_status'].shape == torch.Size([128])
    assert lit_model.logged_results['project_id'].shape == torch.Size([1])

if __name__ == '__main__':
    test_shared_eval_for_checking(lit_model)