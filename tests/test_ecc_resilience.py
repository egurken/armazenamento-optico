"""
Teste de resiliência física: Reed-Solomon + Entrelaçamento 2D contra manchas e danos físicos.
Validação do formato limpo e legível do cabeçalho em QR Code.
"""

import unittest
import os
import json
from datasheet.ecc import calculate_rs_params, encode_rs_interleaved, decode_rs_interleaved
from datasheet.metadata import PageMetadata


class TestECCResilience(unittest.TestCase):

    def test_rs_params(self):
        # 10% de redundância
        k10, nsym10 = calculate_rs_params(10)
        self.assertGreater(k10, 220)
        self.assertEqual(k10 + nsym10, 255)

        # 20% de redundância (padrão recomendado)
        k20, nsym20 = calculate_rs_params(20)
        self.assertGreater(k20, 195)
        self.assertEqual(k20 + nsym20, 255)

        # 30% de redundância
        k30, nsym30 = calculate_rs_params(30)
        self.assertGreater(k30, 170)
        self.assertEqual(k30 + nsym30, 255)

    def test_2d_interleaved_burst_damage_recovery(self):
        """
        Simula uma mancha contínua (dano em rajada/bloco físico de pixels)
        cobrindo ~10% do total de bytes.
        Sem entrelaçamento, destruiria blocos inteiros.
        Com entrelaçamento 2D, o dano é diluído e 100% recuperado.
        """
        original_data = os.urandom(8 * 1024)  # 8 KB de dados
        ecc_percent = 25  # 25% de proteção

        encoded = encode_rs_interleaved(original_data, ecc_percent=ecc_percent)
        total_len = len(encoded)

        # Simula mancha/rasgo cobrindo uma fatia contínua de 10% dos bytes (rajada física)
        corrupted = bytearray(encoded)
        stain_size = int(total_len * 0.10)
        stain_start = total_len // 3
        corrupted[stain_start:stain_start + stain_size] = b"\x00" * stain_size

        # Decodifica e recupera
        recovered = decode_rs_interleaved(bytes(corrupted), len(original_data), ecc_percent=ecc_percent)
        self.assertEqual(original_data, recovered)

    def test_qr_code_json_format(self):
        """
        Verifica que o QR Code utiliza chaves claras, autoexplicativas e legíveis por humanos.
        """
        meta = PageMetadata(
            file_id="42EDB0DDA2",
            page=1,
            pages=1,
            mode="color16",
            copies=5,
            ecc_percent=20,
            grid_cols=720,
            grid_rows=400,
            cell_px=3,
            file_name="documento.pdf",
            raw_size=25037,
            comp_size=13160,
            full_sha256="42edb0dda2233eb7a30a0083415bfd435859ee457b06016a7048b065d7cba5d7",
            chunk_len=13160,
            chunk_sha256="123456"
        )

        json_str = meta.to_json()
        d = json.loads(json_str)

        # Chaves explícitas exigidas pelo usuário
        required_keys = ["id", "page", "total_pages", "mode", "cell_size", "ecc_percent", "filename", "raw_size", "compressed_size", "sha256"]
        for key in required_keys:
            self.assertIn(key, d, f"Chave legível '{key}' deve estar presente no QR Code")

        # Decodificação reversa
        parsed = PageMetadata.from_json(json_str)
        self.assertEqual(parsed.file_id, "42EDB0DDA2")
        self.assertEqual(parsed.ecc_percent, 20)
        self.assertEqual(parsed.file_name, "documento.pdf")


if __name__ == "__main__":
    unittest.main()
