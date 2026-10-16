"""
Módulo de paletas de cores, modulação e demodulação de símbolos.
Calibrado com controle de carga hídrica (Ink Limiting) para impressoras Tanque de Tinta
(como Epson EcoTank L4260), reduzindo o espalhamento capilar nas cores secundárias/escuras
sem perder separação espectral.
Suporta:
- gray4: 4 tons de cinza (2 bits/célula)
- color8: 8 cores dos cantos do cubo RGB/CMY com alívio de carga CMY (3 bits/célula)
- color16: 16 cores otimizadas com dupla luminância (4 bits/célula)
- color32: 32 cores calibradas (5 bits/célula)
"""

from typing import List, Tuple
import cv2
import numpy as np

# Paleta gray4: 00=16, 01=96, 10=176, 11=240
PALETTE_GRAY4 = np.array([
    [16, 16, 16],
    [96, 96, 96],
    [176, 176, 176],
    [240, 240, 240]
], dtype=np.uint8)

# Paleta color8: 8 vértices RGB/CMY com Ink Limiting nas secundárias (R, G, B)
# Evita 200% de carga líquida CMY no papel sulfite em células de 3px
PALETTE_COLOR8 = np.array([
    [16, 16, 16],      # 000: Preto (usa tinta preta pigmentada K)
    [42, 46, 242],     # 001: Azul (alívio de carga C+M para evitar sangramento)
    [38, 232, 42],     # 010: Verde (alívio de carga C+Y)
    [22, 236, 242],    # 011: Ciano (canal C puro, baixo espalhamento)
    [242, 42, 38],     # 100: Vermelho (alívio de carga M+Y)
    [240, 34, 236],    # 101: Magenta (canal M puro)
    [245, 240, 24],    # 110: Amarelo (canal Y puro)
    [248, 248, 248]    # 111: Branco
], dtype=np.uint8)

# Paleta color16: 8 matizes em baixa luminância (com Ink Limiting) + 8 matizes em alta luminância
PALETTE_COLOR16 = np.array([
    # Baixa Luminância (Escuros / 0..7) - piso elevado para evitar encharcamento CMYK
    [16, 16, 16],       # 0: Preto
    [32, 38, 155],      # 1: Azul escuro
    [32, 150, 38],      # 2: Verde escuro
    [28, 150, 152],     # 3: Ciano escuro
    [155, 36, 32],      # 4: Vermelho escuro
    [152, 32, 152],     # 5: Magenta escuro
    [150, 148, 30],     # 6: Amarelo escuro / Oliva
    [112, 112, 112],    # 7: Cinza médio
    # Alta Luminância (Claros / 8..15)
    [182, 182, 182],    # 8: Cinza claro
    [45, 78, 245],      # 9: Azul brilhante
    [42, 232, 50],      # 10: Verde brilhante
    [35, 235, 245],     # 11: Ciano brilhante
    [245, 48, 40],      # 12: Vermelho brilhante
    [245, 45, 245],     # 13: Magenta brilhante
    [245, 238, 35],     # 14: Amarelo brilhante
    [248, 248, 248]     # 15: Branco
], dtype=np.uint8)

# Paleta color32: 32 cores calibradas (5 bits/célula) com máxima distância perceptual e Ink Limiting
PALETTE_COLOR32 = np.array([
    # 0: Preto
    [16, 16, 16],
    # Escuros / Baixa Luminância (1..9) - piso suavizado para jato de tinta
    [152, 34, 34],      # 1: Vermelho escuro
    [160, 84, 28],      # 2: Laranja escuro / Marrom
    [145, 145, 28],     # 3: Oliva / Amarelo escuro
    [34, 150, 34],      # 4: Verde escuro
    [28, 145, 118],     # 5: Teal escuro
    [28, 145, 150],     # 6: Ciano escuro
    [34, 42, 160],      # 7: Azul escuro
    [100, 32, 155],     # 8: Roxo escuro
    [155, 32, 145],     # 9: Vinho
    # Médios / Saturados (10..18)
    [238, 42, 42],      # 10: Vermelho
    [242, 130, 32],     # 11: Laranja
    [242, 228, 28],     # 12: Amarelo
    [135, 232, 32],     # 13: Lima
    [36, 224, 45],      # 14: Verde
    [34, 224, 232],     # 15: Ciano
    [42, 86, 245],      # 16: Azul
    [155, 50, 245],     # 17: Roxo
    [242, 45, 224],     # 18: Magenta
    # Claros / Pastéis (19..27)
    [250, 155, 170],    # 19: Rosa pastel
    [250, 188, 140],    # 20: Pêssego pastel
    [250, 245, 145],    # 21: Amarelo pastel
    [155, 250, 170],    # 22: Menta pastel
    [155, 245, 250],    # 23: Aqua pastel
    [145, 194, 255],    # 24: Azul bebê
    [198, 165, 255],    # 25: Lavanda
    [250, 160, 245],    # 26: Orquídea
    [204, 224, 145],    # 27: Verde chá
    # Escala de cinza neutra (28..31)
    [78, 78, 78],       # 28: Chumbo
    [138, 138, 138],    # 29: Cinza médio
    [198, 198, 198],    # 30: Cinza claro
    [250, 250, 250]     # 31: Branco puro (usado também para preenchimento limpo)
], dtype=np.uint8)


def get_palette(mode: str) -> np.ndarray:
    if mode == "gray4":
        return PALETTE_GRAY4
    elif mode == "color8":
        return PALETTE_COLOR8
    elif mode == "color16":
        return PALETTE_COLOR16
    elif mode == "color32":
        return PALETTE_COLOR32
    else:
        raise ValueError(f"Modo desconhecido: {mode}")


def get_bits_per_cell(mode: str) -> int:
    if mode == "gray4":
        return 2
    elif mode == "color8":
        return 3
    elif mode == "color16":
        return 4
    elif mode == "color32":
        return 5
    else:
        raise ValueError(f"Modo desconhecido: {mode}")


def bytes_to_symbols(data: bytes, mode: str) -> List[int]:
    """
    Converte bytes em lista de símbolos inteiros de acordo com o modo.
    - gray4: 2 bits por símbolo (4 símbolos/byte, MSB first)
    - color8: 3 bits por símbolo
    - color16: 4 bits por símbolo (2 símbolos/byte, MSB first)
    - color32: 5 bits por símbolo
    """
    symbols = []
    if mode == "gray4":
        for b in data:
            symbols.append((b >> 6) & 0x03)
            symbols.append((b >> 4) & 0x03)
            symbols.append((b >> 2) & 0x03)
            symbols.append(b & 0x03)
    elif mode == "color16":
        for b in data:
            symbols.append((b >> 4) & 0x0F)
            symbols.append(b & 0x0F)
    elif mode == "color8":
        bit_buffer = 0
        bit_count = 0
        for b in data:
            bit_buffer = (bit_buffer << 8) | b
            bit_count += 8
            while bit_count >= 3:
                shift = bit_count - 3
                sym = (bit_buffer >> shift) & 0x07
                symbols.append(sym)
                bit_count -= 3
                bit_buffer &= (1 << bit_count) - 1
        if bit_count > 0:
            sym = (bit_buffer << (3 - bit_count)) & 0x07
            symbols.append(sym)
    elif mode == "color32":
        bit_buffer = 0
        bit_count = 0
        for b in data:
            bit_buffer = (bit_buffer << 8) | b
            bit_count += 8
            while bit_count >= 5:
                shift = bit_count - 5
                sym = (bit_buffer >> shift) & 0x1F
                symbols.append(sym)
                bit_count -= 5
                bit_buffer &= (1 << bit_count) - 1
        if bit_count > 0:
            sym = (bit_buffer << (5 - bit_count)) & 0x1F
            symbols.append(sym)
    else:
        raise ValueError(f"Modo não suportado: {mode}")
    return symbols


def symbols_to_bytes(symbols: List[int], mode: str, total_bytes: int) -> bytes:
    """
    Reconstrói o fluxo de bytes original a partir da lista de símbolos.
    """
    out = bytearray()
    if mode == "gray4":
        for i in range(0, len(symbols) - 3, 4):
            if len(out) >= total_bytes:
                break
            b = ((symbols[i] & 0x03) << 6) | \
                ((symbols[i + 1] & 0x03) << 4) | \
                ((symbols[i + 2] & 0x03) << 2) | \
                (symbols[i + 3] & 0x03)
            out.append(b)
    elif mode == "color16":
        for i in range(0, len(symbols) - 1, 2):
            if len(out) >= total_bytes:
                break
            b = ((symbols[i] & 0x0F) << 4) | (symbols[i + 1] & 0x0F)
            out.append(b)
    elif mode == "color8":
        bit_buffer = 0
        bit_count = 0
        for s in symbols:
            bit_buffer = (bit_buffer << 3) | (s & 0x07)
            bit_count += 3
            while bit_count >= 8:
                if len(out) >= total_bytes:
                    break
                shift = bit_count - 8
                b = (bit_buffer >> shift) & 0xFF
                out.append(b)
                bit_count -= 8
                bit_buffer &= (1 << bit_count) - 1
            if len(out) >= total_bytes:
                break
    elif mode == "color32":
        bit_buffer = 0
        bit_count = 0
        for s in symbols:
            bit_buffer = (bit_buffer << 5) | (s & 0x1F)
            bit_count += 5
            while bit_count >= 8:
                if len(out) >= total_bytes:
                    break
                shift = bit_count - 8
                b = (bit_buffer >> shift) & 0xFF
                out.append(b)
                bit_count -= 8
                bit_buffer &= (1 << bit_count) - 1
            if len(out) >= total_bytes:
                break
    return bytes(out)


def classify_color8_hierarchical(pixels: np.ndarray, reference_palette: np.ndarray) -> np.ndarray:
    """
    Classificador hierárquico robusto para o modo color8 baseado em espaço HSV e canais discriminantes:
    - Neutraliza variações de ganho de ponto (dot gain) e sangria de tinta entre células adjacentes.
    - Separa Preto (luminância mínima) e Branco (baixa saturação, alta luminância).
    - Separa Amarelo e Verde pelos intervalos angulares de Matiz (Hue).
    - Separa Ciano vs Azul pela resposta diferencial (G - 0.5 * B) calibrada pelo swatch.
    - Separa Vermelho vs Magenta pela resposta diferencial (B - G) calibrada pelo swatch.
    - Fallback para espaço perceptual CIELAB para células fora dos limites convencionais.
    """
    pix_f = pixels.astype(np.float32)
    pix_u8 = np.clip(pixels, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(pix_u8[None, :, :], cv2.COLOR_RGB2HSV)[0]
    h = hsv[:, 0].astype(np.float32) * 2.0  # 0..360
    s = hsv[:, 1].astype(np.float32) / 255.0
    v = hsv[:, 2].astype(np.float32) / 255.0

    ref_f = reference_palette.astype(np.float32)
    K, B_c, G_c, C_c, R_c, M_c, Y_c, W_c = ref_f

    # Limiar dinâmico para Preto
    k_v = np.max(K) / 255.0
    min_chroma_v = min([np.max(c) for c in [B_c, G_c, C_c, R_c, M_c, Y_c]]) / 255.0
    v_black_th = (k_v + min_chroma_v) * 0.45

    # Limiares discriminantes calibrados pelos swatches observados
    th_cyan_blue = ((B_c[1] - 0.5 * B_c[2]) + (C_c[1] - 0.5 * C_c[2])) / 2.0
    th_mag_red = ((R_c[2] - R_c[1]) + (M_c[2] - M_c[1])) / 2.0 * 0.85

    pred = np.full(len(pixels), -1, dtype=int)

    # 1. Amarelo (6): H em [25, 85]
    is_yellow = (h >= 25) & (h < 85) & (s > 0.18) & (v > v_black_th)
    pred[is_yellow] = 6

    # 2. Verde (2): H em [85, 165]
    is_green = (h >= 85) & (h < 165) & (s > 0.12) & (v > v_black_th)
    pred[is_green] = 2

    # 3. Ciano (3) vs Azul (1): H em [165, 250]
    is_cyan_blue = (h >= 165) & (h < 250) & (s > 0.18) & (v > v_black_th)
    cyan_metric = pix_f[:, 1] - 0.5 * pix_f[:, 2]
    pred[is_cyan_blue & (cyan_metric >= th_cyan_blue)] = 3
    pred[is_cyan_blue & (cyan_metric < th_cyan_blue)] = 1

    # 4. Vermelho (4) vs Magenta (5): H em [250, 360) ou [0, 25)
    is_red_mag = ((h >= 250) | (h < 25)) & (s > 0.18) & (v > v_black_th)
    mag_metric = pix_f[:, 2] - pix_f[:, 1]
    pred[is_red_mag & (mag_metric >= th_mag_red)] = 5
    pred[is_red_mag & (mag_metric < th_mag_red)] = 4

    # 5. Branco (7): Baixa saturação e alta luminância
    w_v = np.max(W_c) / 255.0
    v_white_th = max(0.40, w_v * 0.78)
    is_white = (s <= 0.20) & (v >= v_white_th)
    pred[is_white] = 7

    # 6. Preto (0): Baixíssima luminância
    is_black = (v <= v_black_th)
    pred[is_black] = 0

    # Fallback para qualquer célula não mapeada: distância ponderada em CIELAB
    unassigned = (pred == -1)
    if np.any(unassigned):
        pix_lab = cv2.cvtColor(pix_u8[unassigned][None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        pal_lab = cv2.cvtColor(reference_palette[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        diff = pix_lab[:, None, :] - pal_lab[None, :, :]
        nearest = np.argmin(np.sum(diff**2, axis=2), axis=1)
        pred[unassigned] = nearest

    return pred


def classify_pixels(pixels: np.ndarray, reference_palette: np.ndarray, mode: str = "") -> np.ndarray:
    """
    Classifica cada pixel ao símbolo correspondente mais próximo na paleta de referência.
    - Para color8: utiliza classificação hierárquica por discriminação espectral (HSV + canais diferenciais).
    - Para color16/color32: utiliza o espaço de cor perceptivo CIELAB (L*a*b*).
    - Para gray4: utiliza distância euclidiana simples.
    """
    if mode == "color8" or (not mode and len(reference_palette) == 8):
        try:
            return classify_color8_hierarchical(pixels, reference_palette)
        except Exception:
            pass

    if len(reference_palette) > 4:
        try:
            pix_u8 = np.clip(pixels, 0, 255).astype(np.uint8)
            pal_u8 = np.clip(reference_palette, 0, 255).astype(np.uint8)

            pix_lab = cv2.cvtColor(pix_u8[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
            pal_lab = cv2.cvtColor(pal_u8[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)

            # Peso cromático maior para neutralizar sombras e brilho
            weights = np.array([0.7, 1.8, 1.8], dtype=np.float32)
            diff = pix_lab[:, None, :] - pal_lab[None, :, :]
            weighted_sq = np.sum((diff * weights) ** 2, axis=2)
            return np.argmin(weighted_sq, axis=1)
        except Exception:
            pass

    diff = pixels.astype(np.float32)[:, None, :] - reference_palette.astype(np.float32)[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.argmin(dist_sq, axis=1)
