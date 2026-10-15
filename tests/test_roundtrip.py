"""
Teste de ponta a ponta (Roundtrip):
Arquivo -> Codificação (PNG + PDF) -> Decodificação via OpenCV e QR Code -> Validação de SHA-256.
"""

import unittest
import os
import shutil
import tempfile
import cv2
import numpy as np

from datasheet.config import MODE_GRAY4, MODE_COLOR8, MODE_COLOR16, MODE_COLOR32
from datasheet.encoder import encode_file
from datasheet.decoder import decode_files


class TestRoundtrip(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="datasheet_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_roundtrip_gray4(self):
        # 1. Cria arquivo de teste
        input_path = os.path.join(self.test_dir, "input_gray.txt")
        test_content = b"DataSheet Gray4 Optical Storage Full Pipeline Verification Test!" * 100
        with open(input_path, "wb") as f:
            f.write(test_content)

        png_dir = os.path.join(self.test_dir, "pngs_gray")
        pdf_path = os.path.join(self.test_dir, "output_gray.pdf")

        # 2. Codifica
        encode_file(
            input_path=input_path,
            output_pdf_path=pdf_path,
            output_png_dir=png_dir,
            mode=MODE_GRAY4,
            copies=5,
            show_text=True
        )

        png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")]
        self.assertGreater(len(png_files), 0)

        # 3. Decodifica a partir dos PNGs
        recovered_path = os.path.join(self.test_dir, "recovered_gray.txt")
        recovered_data, orig_fn = decode_files(png_files, output_path=recovered_path)

        self.assertEqual(orig_fn, "input_gray.txt")
        self.assertEqual(test_content, recovered_data)

    def test_roundtrip_color8(self):
        input_path = os.path.join(self.test_dir, "input_color8.bin")
        test_content = os.urandom(12 * 1024)
        with open(input_path, "wb") as f:
            f.write(test_content)

        png_dir = os.path.join(self.test_dir, "pngs_color8")
        encode_file(
            input_path=input_path,
            output_png_dir=png_dir,
            mode=MODE_COLOR8,
            copies=5,
            show_text=True
        )

        png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")]
        recovered_path = os.path.join(self.test_dir, "recovered_color8.bin")
        recovered_data, _ = decode_files(png_files, output_path=recovered_path)

        self.assertEqual(test_content, recovered_data)

    def test_roundtrip_color16(self):
        input_path = os.path.join(self.test_dir, "input_color16.bin")
        test_content = os.urandom(15 * 1024)
        with open(input_path, "wb") as f:
            f.write(test_content)

        png_dir = os.path.join(self.test_dir, "pngs_color16")
        encode_file(
            input_path=input_path,
            output_png_dir=png_dir,
            mode=MODE_COLOR16,
            copies=5,
            show_text=True
        )

        png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")]
        recovered_path = os.path.join(self.test_dir, "recovered_color16.bin")
        recovered_data, _ = decode_files(png_files, output_path=recovered_path)

        self.assertEqual(test_content, recovered_data)

    def test_roundtrip_from_pdf(self):
        input_path = os.path.join(self.test_dir, "input_pdf_test.txt")
        test_content = b"DataSheet PDF Consolidated Direct Decoding Verification Test!" * 80
        with open(input_path, "wb") as f:
            f.write(test_content)

        pdf_path = os.path.join(self.test_dir, "document.pdf")
        encode_file(
            input_path=input_path,
            output_pdf_path=pdf_path,
            mode=MODE_COLOR8,
            copies=5,
            show_text=True
        )

        recovered_path = os.path.join(self.test_dir, "recovered_from_pdf.txt")
        recovered_data, orig_fn = decode_files([pdf_path], output_path=recovered_path)

        self.assertEqual(orig_fn, "input_pdf_test.txt")
        self.assertEqual(test_content, recovered_data)

    def test_roundtrip_color32(self):
        input_path = os.path.join(self.test_dir, "input_color32.bin")
        test_content = os.urandom(20 * 1024)  # 20 KB
        with open(input_path, "wb") as f:
            f.write(test_content)

        png_dir = os.path.join(self.test_dir, "pngs_color32")
        encode_file(
            input_path=input_path,
            output_png_dir=png_dir,
            mode=MODE_COLOR32,
            cell_size=3,
            ecc_percent=20,
            show_text=True
        )

        png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")]
        recovered_path = os.path.join(self.test_dir, "recovered_color32.bin")
        recovered_data, _ = decode_files(png_files, output_path=recovered_path)

        self.assertEqual(test_content, recovered_data)

    def test_roundtrip_cell_size_4(self):
        input_path = os.path.join(self.test_dir, "input_cell4.txt")
        test_content = b"DataSheet Robust Cell Size 4px Verification Payload!" * 50
        with open(input_path, "wb") as f:
            f.write(test_content)

        png_dir = os.path.join(self.test_dir, "pngs_cell4")
        encode_file(
            input_path=input_path,
            output_png_dir=png_dir,
            mode=MODE_COLOR16,
            cell_size=4,
            ecc_percent=20,
            show_text=True
        )

        png_files = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.endswith(".png")]
        recovered_path = os.path.join(self.test_dir, "recovered_cell4.txt")
        recovered_data, _ = decode_files(png_files, output_path=recovered_path)

        self.assertEqual(test_content, recovered_data)


if __name__ == "__main__":
    unittest.main()
