"""
Módulo de redundância e correção de erros via votação de maioria bit a bit.
Permite configurar entre 3 e 6 cópias (padrão: 5).
"""

from typing import List


def apply_repetition(data: bytes, copies: int) -> bytes:
    """
    Replica o bloco de dados o número especificado de cópias consecutivas.
    """
    if not (3 <= copies <= 6):
        raise ValueError("O número de cópias deve estar entre 3 e 6.")
    return data * copies


def majority_vote(repeated_data: bytes, original_len: int, copies: int) -> bytes:
    """
    Aplica algoritmo de votação de maioria bit a bit entre as cópias.
    Para cada bit de cada byte, determina o valor que ocorre na maioria das cópias.
    """
    if not (3 <= copies <= 6):
        raise ValueError("O número de cópias deve estar entre 3 e 6.")

    threshold = (copies + 1) // 2
    result = bytearray(original_len)

    # Coleta os bytes de cada cópia
    copy_slices = []
    for c in range(copies):
        start = c * original_len
        end = start + original_len
        copy_slices.append(repeated_data[start:end])

    for i in range(original_len):
        byte_val = 0
        for bit in range(8):
            mask = 1 << bit
            ones_count = 0
            for c in range(copies):
                if i < len(copy_slices[c]):
                    if copy_slices[c][i] & mask:
                        ones_count += 1
            if ones_count >= threshold:
                byte_val |= mask
        result[i] = byte_val

    return bytes(result)
