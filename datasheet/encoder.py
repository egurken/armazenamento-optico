"""
Módulo codificador e gerador de páginas A4 e PDF consolidado.
Inclui:
1. Alívio nos 4 Cantos da Célula (Corner Relief) para evitar poças de tinta nas diagonais mantendo 100% da capacidade em 3px.
2. Barra de calibração espectral com micro-células reais (simula exatamente o ganho de ponto da grade de 3px).
3. Desativação de interpolação (/Interpolate false) no PDF e metadados de 300 DPI.
"""

import os
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw

from datasheet.config import (
    PageLayout,
    PAGE_WIDTH_A4,
    PAGE_HEIGHT_A4,
    HEADER_MARGIN_TOP_PX,
    CALIBRATION_MARGIN_TOP_PX,
    CALIBRATION_STRIP_HEIGHT_PX,
    DEFAULT_ECC_PERCENT,
    DEFAULT_CELL_PX,
    DEFAULT_CELL_GAP,
    MODE_GRAY4
)
from datasheet.palette import get_palette, bytes_to_symbols
from datasheet.metadata import PageMetadata, generate_qr_image, render_header_text
from datasheet.fiducials import render_page_fiducials
from datasheet.chunker import split_file_to_chunks


def render_isolated_grid(
    grid_rgb: np.ndarray,
    cell_px: int,
    cell_gap: int = 0,
    corner_relief: float = 0.35
) -> Image.Image:
    """
    Escala a matriz de cores (rows, cols, 3) para (rows * cell_px, cols * cell_px, 3).
    - Em modo padrão (cell_gap == 0 e cell_px >= 3), aplica o Alívio de Quinas (Corner Relief):
      mantém o centro da célula (lido pelo decoder) e a cruz '+' com 100% da cor pura,
      mas alivia levemente a carga de tinta apenas nos 4 cantos diagonais de cada célula.
      Isso corta o acúmulo de tinta onde 4 células se encontram sem perder 1 único byte de capacidade.
    - Se cell_gap > 0 for solicitado explicitamente, aplica borda branca entre células.
    """
    rows, cols = grid_rgb.shape[:2]
    if cell_px <= 1:
        return Image.fromarray(grid_rgb, mode="RGB")

    # Expande cada célula para (rows, cell_px, cols, cell_px, 3)
    expanded = np.repeat(np.repeat(grid_rgb, cell_px, axis=0), cell_px, axis=1)
    reshaped = expanded.reshape(rows, cell_px, cols, cell_px, 3)

    eff_gap = max(0, min(int(cell_gap), cell_px - 2)) if cell_px >= 3 else 0
    if eff_gap > 0:
        if eff_gap == 1:
            white_indices = [cell_px - 1] if cell_px in (3, 5) else [0]
        else:
            white_indices = [0, cell_px - 1]

        for idx in white_indices:
            reshaped[:, idx, :, :, :] = 255
            reshaped[:, :, :, idx, :] = 255
    elif cell_px >= 3 and corner_relief > 0.0:
        # Alívio nos 4 cantos diagonais (0,0), (0, c-1), (c-1, 0), (c-1, c-1)
        # O pixel central (int(0.5 * cell_px)) e os braços ortogonais (+) permanecem 100% puros!
        alpha = float(max(0.0, min(0.65, corner_relief)))
        c_last = cell_px - 1
        for cy_idx in (0, c_last):
            for cx_idx in (0, c_last):
                corner = reshaped[:, cy_idx, :, cx_idx, :].astype(np.float32)
                reshaped[:, cy_idx, :, cx_idx, :] = np.clip(
                    corner * (1.0 - alpha) + 255.0 * alpha, 0, 255
                ).astype(np.uint8)

    expanded = reshaped.reshape(rows * cell_px, cols * cell_px, 3)
    return Image.fromarray(expanded, mode="RGB")


def render_calibration_strip(
    page_img: Image.Image,
    draw: ImageDraw.ImageDraw,
    mode: str,
    x: int,
    y: int,
    width: int,
    height: int,
    cell_px: int = DEFAULT_CELL_PX,
    cell_gap: int = DEFAULT_CELL_GAP
):
    """
    Desenha a barra de calibração espectral logo abaixo da grade de dados.
    Utiliza micro-células reais do mesmo tamanho (cell_px) e com o mesmo Alívio de Quinas (Corner Relief)
    da grade de dados, fazendo com que a barra de calibração sofra exatamente o mesmo ganho de ponto
    físico da impressora Epson L4260 que as células de dados.
    """
    palette = get_palette(mode)
    num_colors = len(palette)
    swatch_w = width / float(num_colors)

    # Fundo branco limpo e moldura
    draw.rectangle([x, y, x + width, y + height], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    eff_cell = max(2, int(cell_px))
    inner_h = max(eff_cell, ((height - 4) // eff_cell) * eff_cell)
    rows_calib = max(1, inner_h // eff_cell)

    for i in range(num_colors):
        x1 = int(round(x + i * swatch_w))
        x2 = int(round(x + (i + 1) * swatch_w))
        pad_x = 2 if swatch_w >= 16 else 0
        ix1 = min(x2, x1 + pad_x)
        ix2 = max(ix1 + 1, x2 - pad_x)
        avail_w = max(eff_cell, ix2 - ix1)
        cols_calib = max(1, avail_w // eff_cell)

        # Cria matriz de micro-células da cor i e aplica o mesmo render_isolated_grid (Corner Relief)
        patch_cells = np.full((rows_calib, cols_calib, 3), palette[i], dtype=np.uint8)
        patch_pil = render_isolated_grid(patch_cells, eff_cell, cell_gap=cell_gap)
        patch_pil = patch_pil.crop((0, 0, ix2 - ix1, height - 4))
        page_img.paste(patch_pil, (ix1, y + 2))
        if (ix2 - ix1) >= 6:
            draw.rectangle([ix1, y + 2, ix2, y + height - 2], outline=(0, 0, 0), width=1)


def disable_pdf_image_interpolation(pdf_path: str) -> None:
    """
    Desativa o sinalizador '/Interpolate' em todos os objetos de imagem dentro do PDF,
    forçando leitores de PDF e drivers de impressão a renderizarem pixels nítidos
    (Nearest-Neighbor) sem suavização bicúbica.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import NameObject, BooleanObject

        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)

        for page in writer.pages:
            resources = page.get("/Resources")
            if resources and "/XObject" in resources:
                xobjects = resources["/XObject"].get_object()
                for obj_name in xobjects:
                    xobj = xobjects[obj_name].get_object()
                    if xobj.get("/Subtype") == "/Image":
                        xobj[NameObject("/Interpolate")] = BooleanObject(False)

        with open(pdf_path, "wb") as f:
            writer.write(f)
    except Exception:
        pass


def render_page(meta: PageMetadata, chunk_bytes: bytes, layout: PageLayout) -> Image.Image:
    """
    Renderiza uma única folha A4 com fiduciais, cabeçalho, QR Code, grade de dados e calibração.
    Adapta a altura da grade dinamicamente ao tamanho do arquivo (evita desperdício de tinta preta).
    """
    cell_gap = getattr(meta, "cell_gap", getattr(layout, "cell_gap", DEFAULT_CELL_GAP))

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
        show_text=layout.show_text,
        cell_gap=cell_gap
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

    # 5. Escala para a resolução física da célula com Corner Relief (sem perda de capacidade)
    grid_pil = render_isolated_grid(grid_rgb, page_layout.cell_px, page_layout.cell_gap)

    grid_x = page_layout.grid_left
    grid_y = page_layout.grid_top
    page_img.paste(grid_pil, (grid_x, grid_y))

    # Desenha moldura sutil em torno da grade
    draw.rectangle(
        [grid_x - 1, grid_y - 1, grid_x + page_layout.grid_width_px, grid_y + page_layout.grid_height_px],
        outline=(0, 0, 0),
        width=1
    )

    # 6. Desenha a barra de calibração espectral com micro-células reais logo abaixo da grade
    calib_y = grid_y + page_layout.grid_height_px + CALIBRATION_MARGIN_TOP_PX
    render_calibration_strip(
        page_img,
        draw,
        page_layout.mode,
        grid_x,
        calib_y,
        page_layout.grid_width_px,
        CALIBRATION_STRIP_HEIGHT_PX,
        cell_px=page_layout.cell_px,
        cell_gap=page_layout.cell_gap
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
    show_text: bool = True,
    cell_gap: int = DEFAULT_CELL_GAP
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
        show_text=show_text,
        cell_gap=cell_gap
    )

    # Divide em chunks
    chunks = split_file_to_chunks(file_bytes, file_name, layout)

    rendered_pages = []
    for meta, chunk_bytes in chunks:
        page_img = render_page(meta, chunk_bytes, layout)
        rendered_pages.append(page_img)

    # Salva imagens PNG individuais com metadado explícito de 300 DPI
    if output_png_dir:
        os.makedirs(output_png_dir, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        for i, page_img in enumerate(rendered_pages):
            page_num = i + 1
            png_path = os.path.join(output_png_dir, f"{base_name}_page_{page_num:03d}.png")
            page_img.save(png_path, "PNG", dpi=(300, 300))

    # Salva PDF compilado com todas as páginas (usando compressão lossless Flate via img2pdf e /Interpolate false)
    if output_pdf_path:
        out_dir = os.path.dirname(output_pdf_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import io
        import img2pdf
        png_streams = []
        for page_img in rendered_pages:
            buf = io.BytesIO()
            page_img.save(buf, format="PNG", dpi=(300, 300))
            png_streams.append(buf.getvalue())
        a4_layout = img2pdf.get_layout_fun((img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297)))
        pdf_bytes = img2pdf.convert(png_streams, layout_fun=a4_layout)
        with open(output_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        disable_pdf_image_interpolation(output_pdf_path)

    return rendered_pages
