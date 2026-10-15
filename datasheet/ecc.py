"""
Módulo de proteção contra falhas com Reed-Solomon e Entrelaçamento 2D (Interleaving).
Suporta taxa de paridade configurável (%) e distribuição bidimensional dos bytes
para garantir recuperação mesmo sob manchas, rasgos, borrões ou oclusão física da folha.
"""

import math
from typing import Tuple
import numpy as np
import reedsolo


def calculate_rs_params(ecc_percent: int) -> Tuple[int, int]:
    """
    Calcula o número de bytes de paridade (nsym) e bytes de dados (k)
    para o bloco padrão de 255 bytes em GF(2^8).
    ecc_percent: porcentagem de redundância escolhida pelo usuário (ex: 10, 15, 20, 30%).
    """
    ecc_p = max(5, min(50, int(ecc_percent)))
    nsym = max(4, min(240, round(255 * ecc_p / 100)))
    # Garante que nsym seja par para otimizar detecção/correção
    if nsym % 2 != 0:
        nsym += 1
    k = 255 - nsym
    return k, nsym


def encode_rs_interleaved(data: bytes, ecc_percent: int = 20) -> bytes:
    """
    1. Divide os dados em blocos de $k$ bytes.
    2. Aplica código Reed-Solomon $RS(255, k)$ gerando blocos de 255 bytes.
    3. Executa Entrelaçamento 2D (transposição matricial + deslocamento helicoidal)
       para espalhar os símbolos de cada bloco por toda a superfície física da folha.
    """
    k, nsym = calculate_rs_params(ecc_percent)
    rsc = reedsolo.RSCodec(nsym)
    chunk_len = len(data)
    num_blocks = max(1, math.ceil(chunk_len / k))

    # Preenche o último bloco com zeros caso necessário
    padded = data + b'\x00' * (num_blocks * k - chunk_len)

    # 1. Matriz de blocos RS de dimensão (num_blocks, 255)
    matrix = np.zeros((num_blocks, 255), dtype=np.uint8)
    for i in range(num_blocks):
        matrix[i] = rsc.encode(padded[i * k:(i + 1) * k])

    # 2. Entrelaçamento 2D:
    # Transposição para shape (255, num_blocks)
    trans = matrix.T.copy()
    if num_blocks > 1:
        # Deslocamento helicoidal: gira cada linha de símbolo por um passo fixo coprimo
        for s in range(255):
            shift = (s * 17) % num_blocks
            trans[s] = np.roll(trans[s], shift)

    # Vetor final entrelaçado pronto para mapear na grade 2D
    return trans.flatten().tobytes()


def decode_rs_interleaved(interleaved_bytes: bytes, original_len: int, ecc_percent: int = 20) -> bytes:
    """
    1. Executa o desentrelaçamento 2D (desfaz rotação helicoidal e desfaz transposição).
    2. Corrige erros de símbolos em cada bloco Reed-Solomon de 255 bytes.
    3. Reconstitui o fluxo de bytes original e trunca para original_len.
    """
    k, nsym = calculate_rs_params(ecc_percent)
    rsc = reedsolo.RSCodec(nsym)
    num_blocks = max(1, math.ceil(original_len / k))
    expected_len = num_blocks * 255

    if len(interleaved_bytes) < expected_len:
        raise ValueError(
            f"Fluxo entrelaçado incompleto: recebidos {len(interleaved_bytes)} bytes, "
            f"necessários {expected_len} bytes para os {num_blocks} blocos RS."
        )

    # Converte para matriz (255, num_blocks)
    arr = np.frombuffer(interleaved_bytes[:expected_len], dtype=np.uint8).reshape((255, num_blocks)).copy()

    # Desfaz deslocamento helicoidal
    if num_blocks > 1:
        for s in range(255):
            shift = (s * 17) % num_blocks
            arr[s] = np.roll(arr[s], -shift)

    # Desfaz transposição -> shape (num_blocks, 255)
    matrix = arr.T

    # Decodifica cada bloco individualmente com Reed-Solomon
    recovered = bytearray()
    for i in range(num_blocks):
        blk_bytes = matrix[i].tobytes()
        try:
            dec = rsc.decode(blk_bytes)[0]
            recovered.extend(dec)
        except reedsolo.ReedSolomonError as e:
            raise ValueError(
                f"Falha na correção de erros (bloco {i + 1}/{num_blocks}): danos físicos ou "
                f"manchas na folha excederam o limite de proteção de {ecc_percent}% de ECC."
            ) from e

    return bytes(recovered[:original_len])
