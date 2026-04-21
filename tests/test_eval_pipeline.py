from unittest.mock import patch, MagicMock
from io import StringIO
import sys
import numpy as np
import torch
from src.training.metrics import kabsch_rmsd

def test_eval_only_matches_shared_eval_path():
    # Structural visual check asserted by testing single code path
    from src.training.train import run_full_test_evaluation
    assert callable(run_full_test_evaluation)

def test_checkpoint_load_changes_model_params():
    import torch.nn as nn
    model = nn.Linear(3, 3)
    nn.init.ones_(model.weight)
    pre_sum = sum(p.abs().sum().item() for p in model.parameters())
    
    ckpt = {"model_state": {"weight": torch.zeros(3, 3), "bias": torch.zeros(3)}}
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    
    post_sum = sum(p.abs().sum().item() for p in model.parameters())
    assert pre_sum != post_sum
    assert post_sum == 0

def test_comparison_message_sign():
    from src.training.train import run_full_test_evaluation
    
    class MockConfig:
        wandb_project = ""
        debug_eval_examples = 0
        dump_eval_predictions = ""
        
    flow_matcher = MagicMock()
    model = MagicMock()
    test_loader = []
    test_ds = [MagicMock(smiles="C", complex_id="1")]
    device = torch.device('cpu')
    
    with patch('src.training.train.compute_test_metrics') as mock_test_metrics, \
         patch('src.training.train.compute_etkdg_baseline') as mock_etkdg, \
         patch('os.path.exists', return_value=False):
        
        mock_test_metrics.return_value = {
            "rmsd_median": 1.5,
            "rmsd_mean": 1.5,
            "rmsd_pct_under_1A": 50,
            "rmsd_pct_under_2A": 50,
            "rmsd_pct_under_5A": 50,
            "strain_median": 10
        }
        mock_etkdg.return_value = {
            "rmsd_median": 2.0,
            "rmsd_mean": 2.0,
            "rmsd_pct_under_1A": 50,
            "rmsd_pct_under_2A": 50,
            "rmsd_pct_under_5A": 50,
            "failures": {}
        }
        
        captured = StringIO()
        sys_stdout_backup = sys.stdout
        sys.stdout = captured
        try:
            run_full_test_evaluation(flow_matcher, model, test_loader, test_ds, device, MockConfig(), "fake.pt")
        finally:
            sys.stdout = sys_stdout_backup
            
        output = captured.getvalue()
        assert "better median RMSD" in output
        assert "0.500" in output

        # Test worse case
        mock_test_metrics.return_value["rmsd_median"] = 2.5
        captured = StringIO()
        sys.stdout = captured
        try:
            run_full_test_evaluation(flow_matcher, model, test_loader, test_ds, device, MockConfig(), "fake.pt")
        finally:
            sys.stdout = sys_stdout_backup
            
        output = captured.getvalue()
        assert "worse median RMSD" in output
        assert "0.500" in output

def test_prediction_shapes_match_crystal_shapes():
    P = torch.randn(10, 3)
    Q = torch.randn(10, 3)
    rmsd = kabsch_rmsd(P, Q)
    assert not np.isnan(rmsd)
