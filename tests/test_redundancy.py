"""
Testes unitários para redundância configurável (3 a 6 cópias) e votação de maioria.
"""

import unittest
import random
from datasheet.redundancy import apply_repetition, majority_vote


class TestRedundancy(unittest.TestCase):

    def test_repetition_copies_range(self):
        data = b"Testing repetition data."
        for copies in range(3, 7):
            rep = apply_repetition(data, copies)
            self.assertEqual(len(rep), len(data) * copies)
            recovered = majority_vote(rep, len(data), copies)
            self.assertEqual(data, recovered)

    def test_majority_voting_with_bit_errors(self):
        original = b"Robust optical storage on physical paper!"
        for copies in [3, 5, 6]:
            rep = bytearray(apply_repetition(original, copies))

            # Introduz erros em uma ou duas cópias
            # Para copies=5, corrompe até 2 cópias
            max_corrupted_copies = (copies - 1) // 2
            chunk_len = len(original)

            for c in range(max_corrupted_copies):
                start = c * chunk_len
                for j in range(chunk_len):
                    # Inverte 1 a 2 bits
                    rep[start + j] ^= random.choice([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80])

            # A votação por maioria DEVE recuperar 100% dos dados originais
            recovered = majority_vote(bytes(rep), chunk_len, copies)
            self.assertEqual(original, recovered)


if __name__ == "__main__":
    unittest.main()
