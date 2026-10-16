"""
Módulo de metadados, cabeçalho e geração/leitura do QR Code obrigatório.
"""

from dataclasses import dataclass, asdict
import json
import hashlib
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from datasheet.config import (
    UNIVERSAL_INSTRUCTION,
    PAGE_WIDTH_A4,
    QR_CODE_SIZE_PX,
    DEFAULT_GRID_COLS,
    DEFAULT_GRID_ROWS,
    DEFAULT_CELL_PX,
    DEFAULT_CELL_GAP,
    DEFAULT_COPIES,
    DEFAULT_ECC_PERCENT,
    MODE_GRAY4
)


@dataclass
class PageMetadata:
    file_id: str          # Hash curto único de 10 dígitos hexadecimais (ex: A3F8B29C10)
    page: int             # Página atual (1-based)
    pages: int            # Total de páginas
    mode: str             # gray4, color8, color16, color32
    copies: int           # Fator de repetição legado (ex: 5)
    ecc_percent: int      # Porcentagem de paridade Reed-Solomon ECC (ex: 20%)
    grid_cols: int        # Colunas da grade
    grid_rows: int        # Linhas da grade
    cell_px: int          # Tamanho da célula em pixels
    file_name: str        # Nome do arquivo original
    raw_size: int         # Tamanho bruto do arquivo original
    comp_size: int        # Tamanho total compactado via zlib
    full_sha256: str      # SHA-256 do arquivo original completo (64 hex)
    chunk_len: int        # Tamanho em bytes do chunk compactado desta folha
    chunk_sha256: str     # SHA-256 do chunk desta folha (para validação independente)
    cell_gap: int = DEFAULT_CELL_GAP  # Borda branca anti-sangramento entre células (0, 1 ou 2 px)

    def to_json(self) -> str:
        """
        Gera JSON claro, legível e autoexplicativo para humanos e smartphones.
        Sem siglas enigmáticas de 1 letra.
        """
        data = {
            "id": self.file_id,
            "page": self.page,
            "total_pages": self.pages,
            "mode": self.mode,
            "cell_size": self.cell_px,
            "cell_gap": self.cell_gap,
            "ecc_percent": self.ecc_percent,
            "filename": self.file_name,
            "raw_size": self.raw_size,
            "compressed_size": self.comp_size,
            "sha256": self.full_sha256,
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
            "chunk_len": self.chunk_len,
            "chunk_sha256": self.chunk_sha256
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PageMetadata":
        """
        Decodifica metadados com suporte completo tanto ao novo padrão claro
        quanto a formatos legados de versões anteriores.
        """
        d = json.loads(json_str)
        file_id = d.get("id", "")
        page = int(d.get("page", d.get("p", 1)))
        pages = int(d.get("total_pages", d.get("pages", d.get("n", 1))))
        mode = d.get("mode", d.get("m", "gray4"))
        copies = int(d.get("copies", d.get("c", DEFAULT_COPIES)))
        ecc_percent = int(d.get("ecc_percent", d.get("ecc", DEFAULT_ECC_PERCENT)))
        grid_cols = int(d.get("grid_cols", d.get("w", 720)))
        grid_rows = int(d.get("grid_rows", d.get("h", 840)))
        cell_px = int(d.get("cell_size", d.get("cell_px", d.get("s", 3))))
        cell_gap = int(d.get("cell_gap", d.get("gap", DEFAULT_CELL_GAP)))
        file_name = d.get("filename", d.get("file_name", d.get("fn", "documento")))
        raw_size = int(d.get("raw_size", d.get("raw", 0)))
        comp_size = int(d.get("compressed_size", d.get("comp_size", d.get("comp", 0))))
        full_sha256 = d.get("sha256", d.get("sha", ""))
        chunk_len = int(d.get("chunk_len", d.get("clen", 0)))
        chunk_sha256 = d.get("chunk_sha256", d.get("csha", ""))
        if chunk_len == 0 and comp_size > 0 and pages == 1:
            chunk_len = comp_size

        return cls(
            file_id=file_id,
            page=page,
            pages=pages,
            mode=mode,
            copies=copies,
            ecc_percent=ecc_percent,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            cell_px=cell_px,
            file_name=file_name,
            raw_size=raw_size,
            comp_size=comp_size,
            full_sha256=full_sha256,
            chunk_len=chunk_len,
            chunk_sha256=chunk_sha256,
            cell_gap=cell_gap
        )


def compute_file_id(full_sha256: str) -> str:
    """Gera um identificador único de 10 dígitos hexadecimais em maiúsculas a partir do SHA256."""
    return full_sha256[:10].upper()


def generate_qr_image(meta: PageMetadata, box_size: int = 4) -> Image.Image:
    """
    Gera a imagem do QR Code obrigatório para inclusão na página.
    Usa tamanho inteiro de módulos e borda de 4 módulos (quiet zone)
    garantindo leitura 100% confiável no OpenCV e smartphones.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=box_size,
        border=4
    )
    qr.add_data(meta.to_json())
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def render_header_text(draw: ImageDraw.ImageDraw, meta: PageMetadata, left: int, top: int, max_width: int):
    """
    Renderiza o cabeçalho humano/IA legível:
    1. HASH_10 PAGINA/TOTAL (destacado)
    2. Instrução padronizada para re-implementação universal.
    """
    # Nome legível com identificador e página
    fn = meta.file_name
    if len(fn) > 55:
        fn = fn[:52] + "..."
    title_text = f"{fn} | {meta.file_id} ({meta.page}/{meta.pages})"
    instruction_text = UNIVERSAL_INSTRUCTION

    # Tenta carregar fonte padrão ou sans-serif
    try:
        font_title = ImageFont.truetype("arial.ttf", 34)
        font_inst = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_inst = ImageFont.load_default()

    # Desenha título com Nome, Hash de 10 dígitos e página
    draw.text((left, top), title_text, fill=(16, 16, 16), font=font_title)

    # Quebra a instrução em linhas para caber na largura disponível
    words = instruction_text.split(" ")
    lines = []
    curr_line = ""
    for w in words:
        test_line = f"{curr_line} {w}".strip()
        if len(test_line) * 11 > max_width:
            lines.append(curr_line)
            curr_line = w
        else:
            curr_line = test_line
    if curr_line:
        lines.append(curr_line)

    y = top + 52
    for line in lines:
        draw.text((left, y), line, fill=(64, 64, 64), font=font_inst)
        y += 28


def read_qr_from_image(img_bgr: np.ndarray) -> Optional[PageMetadata]:
    """
    Detecta e decodifica o QR code presente na folha:
    1. Utiliza zxing-cpp (leitor ultrarrápido e robusto para fotos reais de smartphone em qualquer rotação).
    2. Fallback para estratégias com cv2.QRCodeDetector.
    """
    # Estratégia 0: zxing-cpp (tenta imagem direta e rotações se necessário)
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(img_bgr)
        if not results:
            for rot_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                r_img = cv2.rotate(img_bgr, rot_code)
                results = zxingcpp.read_barcodes(r_img)
                if results:
                    break

        for r in results:
            try:
                meta = PageMetadata.from_json(r.text)
                return meta
            except Exception:
                continue
    except Exception:
        pass

    detector = cv2.QRCodeDetector()
    h, w = img_bgr.shape[:2]

    # O cabeçalho fica estritamente antes da grade de dados (y < 515)
    header_h = min(h, 515)
    header_crop = img_bgr[:header_h, :]

    # Estratégia 1: ROI direto da posição padrão do QR Code (x=230, y=60, ~340px)
    y1, y2 = 30, min(header_h, 480)
    x1, x2 = 180, min(w, 650)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size > 0:
        try:
            data, _, _ = detector.detectAndDecode(crop)
            if data:
                return PageMetadata.from_json(data)
        except Exception:
            pass

    # Estratégia 2: Cabeçalho completo
    try:
        data, _, _ = detector.detectAndDecode(header_crop)
        if data:
            return PageMetadata.from_json(data)
    except Exception:
        pass

    # Estratégia 3: Busca por contornos do QR code na região do cabeçalho
    try:
        gray_header = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_header, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if 160 <= cw <= 650 and 160 <= ch <= 650 and 0.75 <= cw / float(ch) <= 1.3:
                pad = 20
                px1 = max(0, cx - pad)
                py1 = max(0, cy - pad)
                px2 = min(header_crop.shape[1], cx + cw + pad)
                py2 = min(header_crop.shape[0], cy + ch + pad)
                candidate = header_crop[py1:py2, px1:px2]
                try:
                    data, _, _ = detector.detectAndDecode(candidate)
                    if data:
                        return PageMetadata.from_json(data)
                except Exception:
                    pass
    except Exception:
        pass

    # Estratégia 4: Imagem completa
    try:
        data, _, _ = detector.detectAndDecode(img_bgr)
        if data:
            return PageMetadata.from_json(data)
    except Exception:
        pass

    return None
