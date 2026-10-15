"""
Módulo de marcadores fiduciais nos 4 cantos e retificação geométrica (homografia).
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw

from datasheet.config import (
    PAGE_WIDTH_A4,
    PAGE_HEIGHT_A4,
    FIDUCIAL_SIZE_PX,
    FIDUCIAL_MARGIN_PX
)


def draw_fiducial_marker(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, marker_type: str):
    """
    Desenha um dos 4 marcadores de canto:
    - 'solid': quadrado preto sólido (Top-Left)
    - 'frame_square': moldura quadrada com quadrado central (Top-Right e Bottom-Left)
    - 'frame_dot': moldura quadrada com ponto central (Bottom-Right)
    """
    x1, y1 = x, y
    x2, y2 = x + size, y + size

    if marker_type == "solid":
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    elif marker_type == "frame_square":
        # Borda externa
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        # Anel branco
        w_step = size // 5
        draw.rectangle([x1 + w_step, y1 + w_step, x2 - w_step, y2 - w_step], fill=(255, 255, 255))
        # Centro preto
        draw.rectangle([x1 + 2 * w_step, y1 + 2 * w_step, x2 - 2 * w_step, y2 - 2 * w_step], fill=(0, 0, 0))
    elif marker_type == "frame_dot":
        # Borda externa
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        # Anel branco
        w_step = size // 6
        draw.rectangle([x1 + w_step, y1 + w_step, x2 - w_step, y2 - w_step], fill=(255, 255, 255))
        # Ponto central preto pequeno
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = size // 8
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0))


def render_page_fiducials(pil_img: Image.Image, margin: int = FIDUCIAL_MARGIN_PX, size: int = FIDUCIAL_SIZE_PX):
    """
    Desenha os 4 marcadores nos 4 cantos da página A4.
    """
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # Top-Left (Sólido)
    draw_fiducial_marker(draw, margin, margin, size, "solid")
    # Top-Right (Moldura com quadrado)
    draw_fiducial_marker(draw, w - margin - size, margin, size, "frame_square")
    # Bottom-Left (Moldura com quadrado)
    draw_fiducial_marker(draw, margin, h - margin - size, size, "frame_square")
    # Bottom-Right (Moldura com ponto)
    draw_fiducial_marker(draw, w - margin - size, h - margin - size, size, "frame_dot")


def get_canonical_fiducial_centers(page_w: int = PAGE_WIDTH_A4, page_h: int = PAGE_HEIGHT_A4,
                                   margin: int = FIDUCIAL_MARGIN_PX, size: int = FIDUCIAL_SIZE_PX) -> np.ndarray:
    """
    Retorna as coordenadas canônicas (X, Y) do centro de cada marcador fiducial na folha de referência:
    [TL, TR, BR, BL]
    """
    half = size / 2.0
    tl = [margin + half, margin + half]
    tr = [page_w - margin - size + half, margin + half]
    br = [page_w - margin - size + half, page_h - margin - size + half]
    bl = [margin + half, page_h - margin - size + half]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_fiducial_centers(img_bgr: np.ndarray, margin_search: int = 225) -> Optional[np.ndarray]:
    """
    Procura os centros dos marcadores fiduciais nos 4 cantos da imagem.
    Primeiro tenta a busca nos 4 cantos delimitados por margin_search.
    Se falhar (ex: fotos de smartphone com margens ou fundos), faz uma busca
    global por contornos pretos quadrados de tamanho coerente com fiduciais (~96px)
    em cada um dos 4 quadrantes da imagem.
    Retorna array shape (4, 2) ordenado como [TL, TR, BR, BL].
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Regiões de busca padrão nos 4 cantos
    quadrants = [
        ("TL", 0, min(h, margin_search), 0, min(w, margin_search)),
        ("TR", 0, min(h, margin_search), max(0, w - margin_search), w),
        ("BR", max(0, h - margin_search), h, max(0, w - margin_search), w),
        ("BL", max(0, h - margin_search), h, 0, min(w, margin_search))
    ]

    centers = []
    found_all_quads = True
    for name, y1, y2, x1, x2 in quadrants:
        crop = gray[y1:y2, x1:x2]
        _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_c = None
        best_score = -1
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)
            aspect = float(rect_w) / max(1, rect_h)
            if 0.7 <= aspect <= 1.3:
                score = area
                if score > best_score:
                    best_score = score
                    best_c = c

        if best_c is not None:
            M = cv2.moments(best_c)
            if M["m00"] > 0:
                centers.append([x1 + M["m10"] / M["m00"], y1 + M["m01"] / M["m00"]])
            else:
                found_all_quads = False
                break
        else:
            found_all_quads = False
            break

    if found_all_quads and len(centers) == 4:
        return np.array(centers, dtype=np.float32)

    # 2. Busca global por marcadores fiduciais nos 4 quadrantes
    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        quad_candidates = {"TL": [], "TR": [], "BR": [], "BL": []}
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1500 or area > 25000:
                continue
            rx, ry, rw, rh = cv2.boundingRect(c)
            aspect = float(rw) / max(1, rh)
            if 0.75 <= aspect <= 1.33:
                extent = area / float(rw * rh)
                if extent < 0.40:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                size_diff = abs(rw - 96) + abs(rh - 96)

                if cx < w * 0.45 and cy < h * 0.45:
                    quad_candidates["TL"].append((size_diff, [cx, cy]))
                elif cx > w * 0.55 and cy < h * 0.45:
                    quad_candidates["TR"].append((size_diff, [cx, cy]))
                elif cx > w * 0.55 and cy > h * 0.55:
                    quad_candidates["BR"].append((size_diff, [cx, cy]))
                elif cx < w * 0.45 and cy > h * 0.55:
                    quad_candidates["BL"].append((size_diff, [cx, cy]))

        res = []
        for q in ["TL", "TR", "BR", "BL"]:
            if not quad_candidates[q]:
                return None
            quad_candidates[q].sort(key=lambda item: item[0])
            res.append(quad_candidates[q][0][1])

        return np.array(res, dtype=np.float32)
    except Exception:
        return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos nas posições [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_crop_paper_contour(img_bgr: np.ndarray) -> np.ndarray:
    """
    Se a imagem for uma fotografia contendo fundo (mesa, piso), localiza o contorno
    de 4 vértices da folha de papel e faz a retificação geométrica prévia.
    """
    h, w = img_bgr.shape[:2]
    scale = 1000.0 / max(w, h)
    small_w, small_h = int(w * scale), int(h * scale)
    small = cv2.resize(img_bgr, (small_w, small_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_pts = None
    min_area = (small_w * small_h) * 0.20
    for c in contours[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > min_area:
            paper_pts = (approx.reshape(4, 2) / scale).astype(np.float32)
            break

    if paper_pts is None:
        return img_bgr

    ordered = order_points(paper_pts)
    (tl, tr, br, bl) = ordered
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    max_w = max(int(w1), int(w2))
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    max_h = max(int(h1), int(h2))

    if max_w > max_h:
        target_w, target_h = 3508, 2480
    else:
        target_w, target_h = 2480, 3508

    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return warped


def rectify_page(img_bgr: np.ndarray, target_w: int = PAGE_WIDTH_A4, target_h: int = PAGE_HEIGHT_A4) -> Optional[np.ndarray]:
    """
    Processa a folha fotografada ou escaneada:
    1. Isola a folha do fundo (mesa/superfície).
    2. Ajusta a rotação automática para vertical (portrait) via QR Code ou orientação da imagem.
    3. Detecta os marcadores fiduciais nos 4 cantos e aplica a Transformada de Perspectiva fina.
    """
    # 1. Pré-retificação de contorno da folha
    page_img = detect_and_crop_paper_contour(img_bgr)

    # 2. Orientação por QR Code (zxing-cpp)
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(page_img)
        if not results:
            for rot_code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                r_img = cv2.rotate(page_img, rot_code)
                results = zxingcpp.read_barcodes(r_img)
                if results:
                    page_img = r_img
                    break

        if results:
            r = results[0]
            if -135 < r.orientation <= -45:
                page_img = cv2.rotate(page_img, cv2.ROTATE_90_CLOCKWISE)
            elif 45 < r.orientation <= 135:
                page_img = cv2.rotate(page_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif abs(r.orientation) > 135:
                page_img = cv2.rotate(page_img, cv2.ROTATE_180)
    except Exception:
        pass

    # Garante proporção vertical (altura >= largura para A4 portrait)
    h, w = page_img.shape[:2]
    if w > h:
        page_img = cv2.rotate(page_img, cv2.ROTATE_90_CLOCKWISE)
        h, w = page_img.shape[:2]

    # 3. Refinamento fino pelos 4 fiduciais
    src_pts = detect_fiducial_centers(page_img, margin_search=280)
    if src_pts is not None:
        dst_pts = get_canonical_fiducial_centers(target_w, target_h)
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(page_img, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC)
        return warped

    # Redimensiona para resolução padrão A4 se necessário (fallback caso não encontre fiduciais)
    if (w, h) != (target_w, target_h):
        page_img = cv2.resize(page_img, (target_w, target_h))

    return page_img
