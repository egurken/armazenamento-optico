"""
Módulo decodificador: leitura de imagem/PDF, detecção de QR, calibração espectral em micro-células,
amostragem de células, cancelamento de interferência lateral 2D (Cross-Talk Cancellation),
votação de maioria e recomposição do arquivo.
"""

import os
import io
import math
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image
from pypdf import PdfReader

from datasheet.config import (
    PageLayout,
    PAGE_WIDTH_A4,
    PAGE_HEIGHT_A4,
    CALIBRATION_MARGIN_TOP_PX,
    CALIBRATION_STRIP_HEIGHT_PX,
    DEFAULT_ECC_PERCENT
)
from datasheet.metadata import PageMetadata, read_qr_from_image
from datasheet.fiducials import rectify_page
from datasheet.palette import get_palette, classify_pixels, symbols_to_bytes
from datasheet.ecc import calculate_rs_params, decode_rs_interleaved
from datasheet.redundancy import majority_vote
from datasheet.chunker import PageAssembler


def apply_unsharp_deconvolution(img_bgr: np.ndarray, amount: float = 1.35, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica máscara de nitidez (Unsharp Mask / desconvolução óptica leve) para reverter
    sangramento físico de tinta (ink bleed) e desfoque de lente entre células adjacentes.
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def cancel_lateral_crosstalk(cell_pixels: np.ndarray, rows: int, cols: int, alpha: float = 0.09) -> np.ndarray:
    """
    Cancelamento de Interferência Lateral 2D (Cross-Talk Cancellation) no nível das células:
    Desconta matematicamente a contaminação espectral (fração alpha) que vazou das 4 células
    ortogonais vizinhas (cima, baixo, esquerda, direita) para o centro da célula (r, c):
      X_est(r, c) = (Y(r, c) - alpha * (Y_up + Y_down + Y_left + Y_right)) / (1 - 4 * alpha)
    """
    if alpha <= 0.0:
        return cell_pixels

    grid = cell_pixels.reshape((rows, cols, 3)).astype(np.float32)
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    neighbors_sum = (
        padded[:-2, 1:-1] +  # cima
        padded[2:, 1:-1] +   # baixo
        padded[1:-1, :-2] +  # esquerda
        padded[1:-1, 2:]     # direita
    )
    denom = max(0.25, 1.0 - 4.0 * alpha)
    decontaminated = (grid - alpha * neighbors_sum) / denom
    return np.clip(decontaminated, 0, 255).astype(np.uint8).reshape(-1, 3)


def sample_calibration_strip(rectified_bgr: np.ndarray, meta: PageMetadata, layout: PageLayout) -> np.ndarray:
    """
    Localiza e extrai as cores reais observadas na fita de calibração impressa.
    Amostra os centros exatos das micro-células dentro de cada bloco da barra de calibração,
    capturando a resposta espectral real das células de tamanho cell_px na impressora.
    Retorna paleta observada de shape (K, 3) em RGB.
    """
    calib_y = layout.grid_top + layout.grid_height_px + CALIBRATION_MARGIN_TOP_PX
    calib_x = layout.grid_left
    calib_w = layout.grid_width_px
    calib_h = CALIBRATION_STRIP_HEIGHT_PX

    strip_crop = rectified_bgr[calib_y:calib_y + calib_h, calib_x:calib_x + calib_w]
    strip_rgb = cv2.cvtColor(strip_crop, cv2.COLOR_BGR2RGB)

    canonical_palette = get_palette(meta.mode)
    num_colors = len(canonical_palette)
    swatch_w = calib_w / float(num_colors)
    eff_cell = max(2, int(meta.cell_px))

    observed = []
    for i in range(num_colors):
        x1 = int(round(i * swatch_w))
        x2 = int(round((i + 1) * swatch_w))
        pad_x = 2 if swatch_w >= 16 else 0
        ix1 = min(x2, x1 + pad_x)
        ix2 = max(ix1 + 1, x2 - pad_x)

        avail_w = max(eff_cell, ix2 - ix1)
        avail_h = max(eff_cell, calib_h - 4)
        cols_c = max(1, avail_w // eff_cell)
        rows_c = max(1, avail_h // eff_cell)

        # Usa os blocos internos se houver >= 3 colunas/linhas, senão todos os centros de micro-células
        r_start, r_end = (1, rows_c - 1) if rows_c >= 3 else (0, rows_c)
        c_start, c_end = (1, cols_c - 1) if cols_c >= 3 else (0, cols_c)

        cy = 2 + ((np.arange(r_start, r_end) + 0.5) * eff_cell).astype(int)
        cx = ix1 + ((np.arange(c_start, c_end) + 0.5) * eff_cell).astype(int)
        cy = np.clip(cy, 0, strip_rgb.shape[0] - 1)
        cx = np.clip(cx, 0, strip_rgb.shape[1] - 1)
        sample_patch = strip_rgb[np.ix_(cy, cx)]

        if sample_patch.size > 0:
            median_color = np.median(sample_patch, axis=(0, 1)).astype(np.uint8)
            observed.append(median_color)
        else:
            observed.append(canonical_palette[i])

    return np.array(observed, dtype=np.uint8)


def sample_grid_cells(rectified_bgr: np.ndarray, meta: PageMetadata, layout: PageLayout, offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Amostra os pixels centrais de cada célula da grade densa de dados,
    com suporte a deslocamento fino (dx, dy) para compensar desvios de impressão/escaneamento.
    Retorna array de shape (rows * cols, 3) em RGB.
    """
    dx, dy = offset
    grid_y = layout.grid_top + dy
    grid_x = layout.grid_left + dx
    grid_w = layout.grid_width_px
    grid_h = layout.grid_height_px

    h_img, w_img = rectified_bgr.shape[:2]
    y1 = max(0, min(h_img - 1, grid_y))
    y2 = max(0, min(h_img, grid_y + grid_h))
    x1 = max(0, min(w_img - 1, grid_x))
    x2 = max(0, min(w_img, grid_x + grid_w))

    grid_crop = rectified_bgr[y1:y2, x1:x2]
    grid_rgb = cv2.cvtColor(grid_crop, cv2.COLOR_BGR2RGB)

    cell_px = meta.cell_px
    centers_y = ((np.arange(meta.grid_rows) + 0.5) * cell_px).astype(int)
    centers_x = ((np.arange(meta.grid_cols) + 0.5) * cell_px).astype(int)

    centers_y = np.clip(centers_y, 0, grid_rgb.shape[0] - 1)
    centers_x = np.clip(centers_x, 0, grid_rgb.shape[1] - 1)

    sampled = grid_rgb[np.ix_(centers_y, centers_x)]  # shape: (rows, cols, 3)
    return sampled.reshape(-1, 3)


def decode_page_image(img_bgr: np.ndarray) -> Tuple[PageMetadata, bytes]:
    """
    Decodifica uma única folha escaneada ou fotografada:
    1. Leitura do QR Code de auto-configuração (antes ou após retificação).
    2. Retificação geométrica de perspectiva (homografia pelos 4 fiduciais).
    3. Amostragem da barra de calibração espectral em micro-células.
    4. Amostragem da grade, cancelamento de interferência lateral (Cross-Talk Cancellation)
       e classificação de símbolos.
    5. Desentrelaçamento 2D e correção de erros Reed-Solomon (ou votação de maioria).
    Retorna: (PageMetadata, chunk_bytes)
    """
    # 1. Tenta ler QR Code na imagem de entrada
    meta = read_qr_from_image(img_bgr)

    # 2. Retificação geométrica por fiduciais
    rectified = rectify_page(img_bgr, target_w=PAGE_WIDTH_A4, target_h=PAGE_HEIGHT_A4)
    if rectified is None:
        rectified = cv2.resize(img_bgr, (PAGE_WIDTH_A4, PAGE_HEIGHT_A4))

    # Se ainda não leu o QR, tenta na imagem retificada
    if meta is None:
        meta = read_qr_from_image(rectified)

    if meta is None:
        raise ValueError("Não foi possível detectar o QR Code de controle nesta folha.")

    layout = PageLayout(
        grid_cols=meta.grid_cols,
        grid_rows=meta.grid_rows,
        cell_px=meta.cell_px,
        mode=meta.mode,
        copies=meta.copies,
        ecc_percent=getattr(meta, 'ecc_percent', DEFAULT_ECC_PERCENT),
        cell_gap=getattr(meta, 'cell_gap', 0)
    )

    # 3. Calibração espectral
    observed_palette = sample_calibration_strip(rectified, meta, layout)

    if getattr(meta, "ecc_percent", 0) > 0:
        # Modo Reed-Solomon com Entrelaçamento 2D
        k, _ = calculate_rs_params(meta.ecc_percent)
        num_blocks = max(1, math.ceil(meta.chunk_len / k))
        total_rs_bytes = num_blocks * 255

        # Candidatos a deslocamento fino (dx, dy) para neutralizar desvios mecânicos
        candidate_offsets = [
            (0, 0), (0, -2), (0, -1), (0, 1),
            (-1, -2), (1, -2), (0, -3), (-1, 0),
            (1, 0), (-1, -1), (1, -1), (-2, -2)
        ]
        last_rs_error = None

        # Passagem 1: Amostragem direta e com Cancelamento de Interferência Lateral 2D (alpha = 0.0, 0.09, 0.14)
        for alpha in (0.0, 0.09, 0.14):
            for offset in candidate_offsets:
                try:
                    cell_pixels = sample_grid_cells(rectified, meta, layout, offset=offset)
                    if alpha > 0.0:
                        cell_pixels = cancel_lateral_crosstalk(
                            cell_pixels, meta.grid_rows, meta.grid_cols, alpha=alpha
                        )
                    symbols = classify_pixels(cell_pixels, observed_palette, mode=meta.mode)
                    raw_interleaved = symbols_to_bytes(symbols.tolist(), meta.mode, total_bytes=total_rs_bytes)
                    chunk_bytes = decode_rs_interleaved(raw_interleaved, original_len=meta.chunk_len, ecc_percent=meta.ecc_percent)
                    return meta, chunk_bytes
                except Exception as e:
                    last_rs_error = e

        # Passagem 2: Desconvolução óptica por máscara de nitidez (Unsharp Mask) + Cross-Talk Cancellation
        sharpened = apply_unsharp_deconvolution(rectified, amount=1.35, sigma=1.0)
        observed_sharp = sample_calibration_strip(sharpened, meta, layout)
        for offset in candidate_offsets[:6]:
            try:
                cell_pixels = sample_grid_cells(sharpened, meta, layout, offset=offset)
                cell_pixels = cancel_lateral_crosstalk(
                    cell_pixels, meta.grid_rows, meta.grid_cols, alpha=0.08
                )
                symbols = classify_pixels(cell_pixels, observed_sharp, mode=meta.mode)
                raw_interleaved = symbols_to_bytes(symbols.tolist(), meta.mode, total_bytes=total_rs_bytes)
                chunk_bytes = decode_rs_interleaved(raw_interleaved, original_len=meta.chunk_len, ecc_percent=meta.ecc_percent)
                return meta, chunk_bytes
            except Exception as e:
                last_rs_error = e

        raise last_rs_error
    else:
        # Modo Legado de Votação por Maioria
        cell_pixels = sample_grid_cells(rectified, meta, layout)
        symbols = classify_pixels(cell_pixels, observed_palette, mode=meta.mode)
        total_repeated_bytes = meta.chunk_len * meta.copies
        repeated_bytes = symbols_to_bytes(symbols.tolist(), meta.mode, total_bytes=total_repeated_bytes)
        chunk_bytes = majority_vote(repeated_bytes, original_len=meta.chunk_len, copies=meta.copies)
        return meta, chunk_bytes


def extract_images_from_pdf(pdf_path: str) -> List[np.ndarray]:
    """
    Extrai imagens das páginas de um arquivo PDF gerado pelo sistema.
    """
    reader = PdfReader(pdf_path)
    images = []
    for page in reader.pages:
        for img_obj in page.images:
            pil_img = Image.open(io.BytesIO(img_obj.data)).convert("RGB")
            arr = np.array(pil_img)
            # RGB para BGR para OpenCV
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            images.append(bgr)
    return images


def load_image_any_path(path: str) -> Optional[np.ndarray]:
    """
    Carrega uma imagem suportando qualquer caractere Unicode/acentos no Windows
    e fotos de altíssima resolução de smartphones (desativando limites de decompression bomb).
    Se a imagem for maior que 3600px, redimensiona proporcionalmente para resolução A4 300 DPI (~3508px).
    """
    try:
        from PIL import Image, ImageOps
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            max_dim = max(w, h)
            if max_dim > 3600:
                scale = 3508.0 / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[Aviso] Falha ao carregar imagem '{path}': {e}")
        return None


def decode_files(input_paths: List[str], output_path: Optional[str] = None) -> Tuple[bytes, str]:
    """
    Processa uma lista de imagens ou PDFs, coleta os chunks em qualquer ordem
    e remonta o arquivo final original com validação do SHA-256.
    """
    assembler = PageAssembler()

    all_images = []
    for p in input_paths:
        if p.lower().endswith(".pdf"):
            all_images.extend(extract_images_from_pdf(p))
        else:
            img = load_image_any_path(p)
            if img is not None:
                all_images.append(img)
            else:
                raise FileNotFoundError(f"Não foi possível abrir a imagem: {p}")

    if not all_images:
        raise ValueError("Nenhuma imagem válida encontrada para decodificação.")

    for i, img_bgr in enumerate(all_images):
        try:
            meta, chunk_bytes = decode_page_image(img_bgr)
            success = assembler.add_page(meta, chunk_bytes)
            if not success:
                print(f"[Aviso] Falha de checksum na página {meta.page}/{meta.pages}. Descartada.")
            else:
                print(f"[OK] Página {meta.page}/{meta.pages} lida com sucesso (ID: {meta.file_id}).")
        except Exception as e:
            print(f"[Erro] Falha ao processar folha {i + 1}: {e}")

    if not assembler.is_complete():
        missing = assembler.get_missing_pages()
        raise RuntimeError(
            f"Documento incompleto! Recebidas {len(assembler.pages_data)} de {assembler.total_pages} páginas.\n"
            f"Faltam as páginas: {missing}"
        )

    raw_data, orig_filename = assembler.assemble()

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(raw_data)
        print(f"[Sucesso] Arquivo original reconstruído e salvo em: {output_path}")

    return raw_data, orig_filename
