"""
Testes unitários para paletas e modulação/demodulação de símbolos.
"""

import unittest
import numpy as np
from datasheet.config import (
    MODE_GRAY4,
    MODE_COLOR8,
    MODE_COLOR16
)
from datasheet.palette import (
    bytes_to_symbols,
    symbols_to_bytes,
    classify_pixels,
    get_palette
)


class TestPalette(unittest.TestCase):

    def test_roundtrip_gray4(self):
        original = b"Hello DataSheet Optical Storage Test 12345!"
        symbols = bytes_to_symbols(original, MODE_GRAY4)
        recovered = symbols_to_bytes(symbols, MODE_GRAY4, len(original))
        self.assertEqual(original, recovered)

    def test_roundtrip_color8(self):
        original = b"A quick brown fox jumps over the lazy dog. 9876543210!"
        symbols = bytes_to_symbols(original, MODE_COLOR8)
        recovered = symbols_to_bytes(symbols, MODE_COLOR8, len(original))
        self.assertEqual(original, recovered)

    def test_roundtrip_color16(self):
        original = b"Testing 16-color high density optical modulation!"
        symbols = bytes_to_symbols(original, MODE_COLOR16)
        recovered = symbols_to_bytes(symbols, MODE_COLOR16, len(original))
        self.assertEqual(original, recovered)

    def test_classify_with_noise(self):
        # Testa a classificação mesmo com ruído gaussiano adicionado
        palette = get_palette(MODE_COLOR8)
        # Cria pixels com leve perturbação (+- 15 nos canais)
        noisy_pixels = palette.astype(np.float32) + np.random.uniform(-15, 15, palette.shape)
        noisy_pixels = np.clip(noisy_pixels, 0, 255).astype(np.uint8)

        classified = classify_pixels(noisy_pixels, palette)
        expected = np.arange(len(palette))
        np.testing.assert_array_equal(classified, expected)


if __name__ == "__main__":
    unittest.main()
