#!/usr/bin/env python3
"""Quick test script to verify the training setup.

This script tests:
1. Model loading
2. Loss functions
3. Forward pass
4. Device availability
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.train_first_step.model_loader import load_tts_models, get_model_size_info
from training.train_first_step.losses import CombinedTTSLoss


def test_device():
    """Test device availability."""
    print("=" * 80)
    print("Testing Device")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    return device


def test_model_loading(device):
    """Test model loading."""
    print("=" * 80)
    print("Testing Model Loading")
    print("=" * 80)
    
    try:
        model = load_tts_models(
            device=device,
            acoustic_decoder_hidden_size=256,
            acoustic_decoder_num_layers=3,
            style_embedding_dim=128,
        )
        print("✓ Model loaded successfully\n")
        
        # Get model size info
        model_info = get_model_size_info(model)
        print("Model Size Information:")
        print(f"  Text Encoder: {model_info['text_encoder']:,} parameters")
        print(f"  Acoustic Decoder: {model_info['acoustic_decoder']:,} parameters")
        print(f"  Style Extractor: {model_info['style_extractor']:,} parameters")
        print(f"  Vocoder: {model_info['vocoder']:,} parameters")
        print(f"  Trainable: {model_info['trainable']:,} parameters")
        print(f"  Total: {model_info['total']:,} parameters\n")
        
        return model
    
    except Exception as e:
        print(f"✗ Error loading model: {e}\n")
        return None


def test_forward_pass(model, device):
    """Test forward pass."""
    print("=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)
    
    batch_size = 4
    max_text_len = 256
    n_mels = 80
    max_mel_len = 512
    
    try:
        model.eval()
        
        # Create dummy inputs
        text_ids = torch.randint(0, 1000, (batch_size, max_text_len)).to(device)
        target_mel = torch.randn(batch_size, n_mels, max_mel_len).to(device)
        
        print(f"Input shapes:")
        print(f"  text_ids: {text_ids.shape}")
        print(f"  target_mel: {target_mel.shape}")
        
        with torch.no_grad():
            predicted_mel, style_embeddings = model(
                text_ids=text_ids,
                target_mel=target_mel,
                use_vocoder=False,
            )
        
        print(f"\nOutput shapes:")
        print(f"  predicted_mel: {predicted_mel.shape}")
        print(f"  style_embeddings: {style_embeddings.shape}")
        print("✓ Forward pass successful\n")
        
        return True
    
    except Exception as e:
        print(f"✗ Error in forward pass: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions(model, device):
    """Test loss functions."""
    print("=" * 80)
    print("Testing Loss Functions")
    print("=" * 80)
    
    batch_size = 4
    max_text_len = 256
    n_mels = 80
    max_mel_len = 512
    
    try:
        criterion = CombinedTTSLoss(
            weight_reconstruction=1.0,
            weight_diversity=0.5,
            diversity_margin=0.1,
        ).to(device)
        
        # Generate dummy predictions and targets
        predicted_mel = torch.randn(batch_size, n_mels, max_mel_len, requires_grad=True).to(device)
        target_mel = torch.randn(batch_size, n_mels, max_mel_len).to(device)
        style_embeddings = torch.randn(batch_size, 128, requires_grad=True).to(device)
        
        # Compute losses
        total_loss, recon_loss, div_loss = criterion(
            predicted_mel=predicted_mel,
            target_mel=target_mel,
            style_embeddings=style_embeddings,
        )
        
        print(f"Loss values:")
        print(f"  Total Loss: {total_loss.item():.6f}")
        print(f"  Reconstruction Loss: {recon_loss.item():.6f}")
        print(f"  Diversity Loss: {div_loss.item():.6f}")
        
        # Test backward pass
        total_loss.backward()
        print(f"\n✓ Loss functions working correctly")
        print(f"  Predicted mel gradients: {predicted_mel.grad is not None}")
        print(f"  Style embeddings gradients: {style_embeddings.grad is not None}\n")
        
        return True
    
    except Exception as e:
        print(f"✗ Error in loss functions: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_structure():
    """Test experiment directory structure creation."""
    print("=" * 80)
    print("Testing Experiment Structure")
    print("=" * 80)
    
    try:
        from training.train_first_step.train import create_experiment_dir
        
        experiment_dir = create_experiment_dir("test_experiment")
        
        print(f"✓ Experiment directory created: {experiment_dir}")
        print(f"  Structure:")
        print(f"    ├─ checkpoints/ {(experiment_dir / 'checkpoints').exists()}")
        print(f"    ├─ tensorboard/ {(experiment_dir / 'tensorboard').exists()}")
        print(f"    └─ logs/ {(experiment_dir / 'logs').exists()}\n")
        
        return True
    
    except Exception as e:
        print(f"✗ Error creating experiment structure: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "First-Step TTS Training - Test Suite" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = {}
    
    # Test device
    device = test_device()
    
    # Test model loading
    model = test_model_loading(device)
    results["model_loading"] = model is not None
    
    # Test forward pass
    if model:
        results["forward_pass"] = test_forward_pass(model, device)
    
    # Test loss functions
    if model:
        results["loss_functions"] = test_loss_functions(model, device)
    
    # Test experiment structure
    results["experiment_structure"] = test_experiment_structure()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    print()
    
    if all_passed:
        print("✓ All tests passed! You're ready to start training.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    print("\nTo start training, run:")
    print("  python src/training/train_first_step/train.py --num-epochs 10 --batch-size 4")
    print()


if __name__ == "__main__":
    main()
