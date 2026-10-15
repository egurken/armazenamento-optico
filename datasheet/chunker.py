"""
Módulo de particionamento multi-folhas (Chunker) e recomposição de páginas (Assembler).
"""

import hashlib
import math
import zlib
from typing import List, Dict, Tuple, Optional

from datasheet.config import (
    PageLayout,
    DEFAULT_COPIES,
    DEFAULT_ECC_PERCENT,
    MODE_GRAY4
)
from datasheet.palette import get_bits_per_cell
from datasheet.ecc import calculate_rs_params, encode_rs_interleaved
from datasheet.redundancy import apply_repetition, majority_vote
from datasheet.metadata import PageMetadata, compute_file_id


def calculate_page_capacity(layout: PageLayout) -> Tuple[int, int]:
    """
    Calcula a capacidade da grade em bytes totais e em bytes úteis do payload.
    Utiliza a taxa de Reed-Solomon configurada (layout.ecc_percent) ou cópias legadas.
    Retorna: (total_grid_bytes, useful_chunk_bytes)
    """
    total_cells = layout.total_cells
    bits_per_cell = get_bits_per_cell(layout.mode)
    total_bits = total_cells * bits_per_cell
    total_grid_bytes = total_bits // 8

    if getattr(layout, 'ecc_percent', 0) > 0:
        k, _ = calculate_rs_params(layout.ecc_percent)
        num_blocks = total_grid_bytes // 255
        useful_chunk_bytes = max(k, num_blocks * k)
    else:
        useful_chunk_bytes = total_grid_bytes // layout.copies
    return total_grid_bytes, useful_chunk_bytes


def calculate_needed_rows(payload_bytes: int, ecc_percent: int, grid_cols: int, mode: str, max_rows: int = 840, min_rows: int = 60, copies: int = 5) -> int:
    """
    Calcula a quantidade exata de linhas necessárias na grade para o payload com Reed-Solomon ou cópias.
    Evita imprimir blocos gigantes pretos ou vazios em arquivos pequenos.
    """
    if ecc_percent > 0:
        k, _ = calculate_rs_params(ecc_percent)
        num_blocks = max(1, math.ceil(payload_bytes / k))
        total_ecc_bytes = num_blocks * 255
    else:
        total_ecc_bytes = payload_bytes * copies

    bits_per_cell = get_bits_per_cell(mode)
    total_bits = total_ecc_bytes * 8
    needed_cells = math.ceil(total_bits / bits_per_cell)
    needed_rows = math.ceil(needed_cells / grid_cols)
    needed_rows = ((needed_rows + 3) // 4) * 4
    return max(min_rows, min(max_rows, needed_rows))


def split_file_to_chunks(data: bytes, file_name: str, layout: PageLayout) -> List[Tuple[PageMetadata, bytes]]:
    """
    Compacta o arquivo de entrada com zlib e divide em $N$ páginas/chunks.
    Aplica Reed-Solomon ECC e Entrelaçamento 2D para proteger contra manchas e rasgos.
    Retorna uma lista de tuplas: (PageMetadata, interleaved_padded_bytes)
    """
    raw_size = len(data)
    full_sha256 = hashlib.sha256(data).hexdigest()
    file_id = compute_file_id(full_sha256)

    # Compacta com nível máximo do zlib
    compressed = zlib.compress(data, level=9)
    comp_size = len(compressed)

    total_grid_bytes, useful_chunk_bytes = calculate_page_capacity(layout)

    # Determina o número de páginas necessárias
    total_pages = max(1, math.ceil(comp_size / useful_chunk_bytes))

    pages_out = []
    for p in range(1, total_pages + 1):
        start = (p - 1) * useful_chunk_bytes
        end = min(comp_size, start + useful_chunk_bytes)
        chunk_data = compressed[start:end]
        chunk_len = len(chunk_data)
        chunk_sha = hashlib.sha256(chunk_data).hexdigest()

        # Ajusta a grade proporcionalmente ao tamanho real do bloco com Reed-Solomon ou repetição
        page_rows = calculate_needed_rows(
            payload_bytes=chunk_len,
            ecc_percent=getattr(layout, 'ecc_percent', 0),
            grid_cols=layout.grid_cols,
            mode=layout.mode,
            max_rows=layout.grid_rows,
            copies=layout.copies
        )

        bits_per_cell = get_bits_per_cell(layout.mode)
        page_total_cells = layout.grid_cols * page_rows
        page_total_grid_bytes = (page_total_cells * bits_per_cell) // 8

        meta = PageMetadata(
            file_id=file_id,
            page=p,
            pages=total_pages,
            mode=layout.mode,
            copies=layout.copies,
            ecc_percent=getattr(layout, 'ecc_percent', 0),
            grid_cols=layout.grid_cols,
            grid_rows=page_rows,
            cell_px=layout.cell_px,
            file_name=file_name,
            raw_size=raw_size,
            comp_size=comp_size,
            full_sha256=full_sha256,
            chunk_len=chunk_len,
            chunk_sha256=chunk_sha
        )

        # 1. Aplica codificação Reed-Solomon ou repetição
        if getattr(layout, 'ecc_percent', 0) > 0:
            interleaved_bytes = encode_rs_interleaved(chunk_data, ecc_percent=layout.ecc_percent)
        else:
            interleaved_bytes = apply_repetition(chunk_data, layout.copies)

        # 2. Preenche o restante com BRANCO (0xFF), deixando a folha limpa
        if len(interleaved_bytes) < page_total_grid_bytes:
            interleaved_bytes = interleaved_bytes + b'\xFF' * (page_total_grid_bytes - len(interleaved_bytes))
        else:
            interleaved_bytes = interleaved_bytes[:page_total_grid_bytes]

        pages_out.append((meta, interleaved_bytes))

    return pages_out


class PageAssembler:
    """
    Coletor e recompositor de páginas multi-folhas.
    Permite receber páginas escaneadas em qualquer ordem e validar integridade.
    """

    def __init__(self):
        self.file_id: Optional[str] = None
        self.total_pages: Optional[int] = None
        self.file_name: Optional[str] = None
        self.full_sha256: Optional[str] = None
        self.raw_size: Optional[int] = None
        self.comp_size: Optional[int] = None
        self.pages_data: Dict[int, bytes] = {}
        self.pages_metadata: Dict[int, PageMetadata] = {}

    def add_page(self, meta: PageMetadata, chunk_data: bytes) -> bool:
        """
        Adiciona um chunk decodificado de uma folha.
        Valida o hash SHA-256 do chunk individual antes de aceitá-lo.
        """
        actual_sha = hashlib.sha256(chunk_data).hexdigest()
        if actual_sha != meta.chunk_sha256:
            return False

        if self.file_id is None:
            self.file_id = meta.file_id
            self.total_pages = meta.pages
            self.file_name = meta.file_name
            self.full_sha256 = meta.full_sha256
            self.raw_size = meta.raw_size
            self.comp_size = meta.comp_size
        else:
            if self.file_id != meta.file_id:
                raise ValueError(f"ID de arquivo divergente! Esperado {self.file_id}, recebido {meta.file_id}")

        self.pages_data[meta.page] = chunk_data
        self.pages_metadata[meta.page] = meta
        return True

    def get_missing_pages(self) -> List[int]:
        if self.total_pages is None:
            return []
        return [p for p in range(1, self.total_pages + 1) if p not in self.pages_data]

    def is_complete(self) -> bool:
        if self.total_pages is None:
            return False
        return len(self.pages_data) == self.total_pages

    def assemble(self) -> Tuple[bytes, str]:
        """
        Junta os chunks em ordem, descomprime via zlib e valida o SHA-256 global.
        Retorna (dados_originais, nome_arquivo).
        """
        if not self.is_complete():
            missing = self.get_missing_pages()
            raise ValueError(f"Documento incompleto! Faltam as páginas: {missing}")

        # Concatena os chunks na ordem de 1 a N
        compressed_full = bytearray()
        for p in range(1, self.total_pages + 1):
            compressed_full.extend(self.pages_data[p])

        # Descomprime o payload
        raw_data = zlib.decompress(bytes(compressed_full))

        # Validação do SHA-256
        actual_sha = hashlib.sha256(raw_data).hexdigest()
        if actual_sha != self.full_sha256:
            raise ValueError(
                f"Falha de integridade SHA-256 global!\n"
                f"Esperado: {self.full_sha256}\n"
                f"Obtido:   {actual_sha}"
            )

        return raw_data, self.file_name or "arquivo_recuperado.bin"
