"""
Módulo codificador e gerador de páginas A4 e PDF consolidado.
"""

import os
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw, JpegImagePlugin

from datasheet.config import (
    PageLayout,
    PAGE_WIDTH_A4,
    PAGE_HEIGHT_A4,
    HEADER_MARGIN_TOP_PX,
    CALIBRATION_MARGIN_TOP_PX,
    CALIBRATION_STRIP_HEIGHT_PX,
    DEFAULT_ECC_PERCENT,
    DEFAULT_CELL_PX,
    MODE_GRAY4
)
from datasheet.palette import get_palette, bytes_to_symbols
from datasheet.metadata import PageMetadata, generate_qr_image, render_header_text
from datasheet.fiducials import render_page_fiducials
from datasheet.chunker import split_file_to_chunks


def render_calibration_strip(draw: ImageDraw.ImageDraw, mode: str, x: int, y: int, width: int, height: int):
    """
    Desenha a barra de calibração espectral obrigatória logo abaixo da grade de dados.
    """
    palette = get_palette(mode)
    num_colors = len(palette)
    swatch_w = width / float(num_colors)

    for i in range(num_colors):
        x1 = int(round(x + i * swatch_w))
        x2 = int(round(x + (i + 1) * swatch_w))
        color = tuple(int(c) for c in palette[i])
        draw.rectangle([x1, y, x2, y + height], fill=color, outline=(0, 0, 0), width=1)

    # Contorno geral da barra
    draw.rectangle([x, y, x + width, y + height], outline=(0, 0, 0), width=2)


def render_page(meta: PageMetadata, chunk_bytes: bytes, layout: PageLayout) -> Image.Image:
    """
    Renderiza uma única folha A4 com fiduciais, cabeçalho, QR Code, grade de dados e calibração.
    Adapta a altura da grade dinamicamente ao tamanho do arquivo (evita desperdício de tinta preta).
    """
    # Layout específico desta folha com altura adaptativa
    page_layout = PageLayout(
        page_w=layout.page_w,
        page_h=layout.page_h,
        grid_cols=meta.grid_cols,
        grid_rows=meta.grid_rows,
        cell_px=meta.cell_px,
        mode=meta.mode,
        copies=meta.copies,
        ecc_percent=meta.ecc_percent,
        show_text=layout.show_text
    )

    # 1. Cria a página em branco (fundo branco)
    page_img = Image.new("RGB", (page_layout.page_w, page_layout.page_h), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)

    # 2. Desenha os 4 marcadores fiduciais nos cantos
    render_page_fiducials(page_img)

    # 3. Desenha o QR Code obrigatório no cabeçalho (sem sobreposição com fiducial ou grade)
    qr_img = generate_qr_image(meta, box_size=4)
    qr_w, qr_h = qr_img.size
    qr_x = 230
    qr_y = 60
    page_img.paste(qr_img, (qr_x, qr_y))

    # 4. Desenha o cabeçalho humano/IA legível (se ativado)
    if page_layout.show_text:
        text_x = qr_x + qr_w + 40
        text_y = qr_y + 20
        max_text_w = (page_layout.page_w - page_layout.grid_left) - text_x
        render_header_text(draw, meta, text_x, text_y, max_text_w)

    palette = get_palette(page_layout.mode)
    white_symbol = len(palette) - 1
    symbols = bytes_to_symbols(chunk_bytes, page_layout.mode)
    expected_cells = page_layout.total_cells
    if len(symbols) < expected_cells:
        symbols.extend([white_symbol] * (expected_cells - len(symbols)))
    else:
        symbols = symbols[:expected_cells]

    symbols_arr = np.array(symbols, dtype=np.int32).reshape((page_layout.grid_rows, page_layout.grid_cols))
    grid_rgb = palette[symbols_arr]  # shape: (rows, cols, 3)

    # Escala para a resolução física de pixels da célula
    grid_pil = Image.fromarray(grid_rgb, mode="RGB")
    if page_layout.cell_px > 1:
        grid_pil = grid_pil.resize(
            (page_layout.grid_width_px, page_layout.grid_height_px),
            Image.Resampling.NEAREST
        )

    grid_x = page_layout.grid_left
    grid_y = page_layout.grid_top
    page_img.paste(grid_pil, (grid_x, grid_y))

    # Desenha moldura sutil em torno da grade
    draw.rectangle(
        [grid_x - 1, grid_y - 1, grid_x + page_layout.grid_width_px, grid_y + page_layout.grid_height_px],
        outline=(0, 0, 0),
        width=1
    )

    # 6. Desenha a barra de calibração espectral logo abaixo da grade adaptada
    calib_y = grid_y + page_layout.grid_height_px + CALIBRATION_MARGIN_TOP_PX
    render_calibration_strip(
        draw,
        page_layout.mode,
        grid_x,
        calib_y,
        page_layout.grid_width_px,
        CALIBRATION_STRIP_HEIGHT_PX
    )

    return page_img


def encode_file(
    input_path: str,
    output_pdf_path: Optional[str] = None,
    output_png_dir: Optional[str] = None,
    mode: str = MODE_GRAY4,
    copies: int = 5,
    ecc_percent: int = DEFAULT_ECC_PERCENT,
    cell_size: int = DEFAULT_CELL_PX,
    show_text: bool = True
) -> List[Image.Image]:
    """
    Codifica um arquivo do disco e gera o PDF consolidado e/ou imagens PNG de cada página.
    """
    with open(input_path, "rb") as f:
        file_bytes = f.read()

    file_name = os.path.basename(input_path)
    layout = PageLayout(
        mode=mode,
        copies=copies,
        ecc_percent=ecc_percent,
        cell_px=cell_size,
        show_text=show_text
    )

    # Divide em chunks
    chunks = split_file_to_chunks(file_bytes, file_name, layout)

    rendered_pages = []
    for meta, chunk_bytes in chunks:
        page_img = render_page(meta, chunk_bytes, layout)
        rendered_pages.append(page_img)

    # Salva imagens PNG individuais se solicitado
    if output_png_dir:
        os.makedirs(output_png_dir, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        for i, page_img in enumerate(rendered_pages):
            page_num = i + 1
            png_path = os.path.join(output_png_dir, f"{base_name}_page_{page_num:03d}.png")
            page_img.save(png_path, "PNG")

    # Salva PDF compilado com todas as páginas (usando compressão lossless Flate via img2pdf)
    if output_pdf_path:
        out_dir = os.path.dirname(output_pdf_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import io
        import img2pdf
        png_streams = []
        for page_img in rendered_pages:
            buf = io.BytesIO()
            page_img.save(buf, format="PNG")
            png_streams.append(buf.getvalue())
        pdf_bytes = img2pdf.convert(png_streams)
        with open(output_pdf_path, "wb") as f:
            f.write(pdf_bytes)

    return rendered_pages
