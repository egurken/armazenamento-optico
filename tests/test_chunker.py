"""
Testes unitários para divisão de arquivos em multi-páginas e remontagem desordenada.
"""

import unittest
import os
import random
from datasheet.config import PageLayout, MODE_GRAY4, MODE_COLOR8, MODE_COLOR16
from datasheet.chunker import split_file_to_chunks, PageAssembler
from datasheet.redundancy import majority_vote


from datasheet.ecc import decode_rs_interleaved


class TestChunker(unittest.TestCase):

    def test_multipage_split_and_out_of_order_assembly_rs(self):
        # Gera um arquivo de teste de 50 KB
        test_payload = os.urandom(50 * 1024)
        file_name = "test_data_rs.bin"

        # Usa layout com capacidade menor por folha para forçar multi-páginas com Reed-Solomon
        layout = PageLayout(
            grid_cols=300,
            grid_rows=300,
            cell_px=3,
            mode=MODE_COLOR8,
            ecc_percent=20
        )

        chunks = split_file_to_chunks(test_payload, file_name, layout)
        self.assertGreater(len(chunks), 1, "Deveria ter gerado mais de 1 página")

        # Simula decodificação de cada chunk usando Reed-Solomon
        decoded_chunks = []
        for meta, raw_bytes in chunks:
            chunk_bytes = decode_rs_interleaved(raw_bytes, meta.chunk_len, meta.ecc_percent)
            decoded_chunks.append((meta, chunk_bytes))

        # Embaralha as páginas para simular escaneamento fora de ordem (ex: 3, 1, 2)
        random.shuffle(decoded_chunks)

        assembler = PageAssembler()
        for meta, chunk_bytes in decoded_chunks:
            added = assembler.add_page(meta, chunk_bytes)
            self.assertTrue(added)

        self.assertTrue(assembler.is_complete())
        recovered_data, recovered_fn = assembler.assemble()

        self.assertEqual(recovered_fn, file_name)
        self.assertEqual(test_payload, recovered_data)

    def test_multipage_split_and_out_of_order_assembly_legacy(self):
        # Gera um arquivo de teste de 50 KB
        test_payload = os.urandom(50 * 1024)
        file_name = "test_data_legacy.bin"

        # Usa layout com repetição simples
        layout = PageLayout(
            grid_cols=300,
            grid_rows=300,
            cell_px=3,
            mode=MODE_COLOR8,
            copies=5,
            ecc_percent=0
        )

        chunks = split_file_to_chunks(test_payload, file_name, layout)
        self.assertGreater(len(chunks), 1, "Deveria ter gerado mais de 1 página")

        # Simula decodificação de cada chunk com majority vote
        decoded_chunks = []
        for meta, repeated_bytes in chunks:
            chunk_bytes = majority_vote(repeated_bytes, meta.chunk_len, meta.copies)
            decoded_chunks.append((meta, chunk_bytes))

        random.shuffle(decoded_chunks)

        assembler = PageAssembler()
        for meta, chunk_bytes in decoded_chunks:
            added = assembler.add_page(meta, chunk_bytes)
            self.assertTrue(added)

        self.assertTrue(assembler.is_complete())
        recovered_data, recovered_fn = assembler.assemble()

        self.assertEqual(recovered_fn, file_name)
        self.assertEqual(test_payload, recovered_data)


if __name__ == "__main__":
    unittest.main()
