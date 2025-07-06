import torch
import unittest

from src.model.transformer import BinocularformerEncoder

class TestBinocularformerEncoder(unittest.TestCase):
    def test_random_input(self):
        batch_size = 2
        seq_len = 16
        d_model = 32
        encoder = BinocularformerEncoder(
            num_layers=2,
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder(x)
        self.assertEqual(output.shape, (batch_size, seq_len + 3, d_model))

if __name__ == "__main__":
    unittest.main()