"""
Configurações e constantes para o sistema DataSheet.
"""

from dataclasses import dataclass
from typing import Tuple

# Modos suportados
MODE_GRAY4 = "gray4"      # 4 tons de cinza (PAM-4, 2 bits/célula)
MODE_COLOR8 = "color8"    # 8 cores do cubo RGB/CMY (3 bits/célula)
MODE_COLOR16 = "color16"  # 16 cores otimizadas em Lab (4 bits/célula)
MODE_COLOR32 = "color32"  # 32 cores calibradas (5 bits/célula - Máxima densidade)

SUPPORTED_MODES = [MODE_GRAY4, MODE_COLOR8, MODE_COLOR16, MODE_COLOR32]
SUPPORTED_CELL_SIZES = [2, 3, 4, 5, 6]

# Redundância (cópias legadas e Reed-Solomon ECC %)
DEFAULT_COPIES = 5
MIN_COPIES = 3
MAX_COPIES = 6

DEFAULT_ECC_PERCENT = 20
MIN_ECC_PERCENT = 5
MAX_ECC_PERCENT = 50

# Resolução A4 em 300 DPI (210 x 297 mm)
PAGE_WIDTH_A4 = 2480
PAGE_HEIGHT_A4 = 3508

# Grade padrão (células)
DEFAULT_GRID_COLS = 720
DEFAULT_GRID_ROWS = 840
DEFAULT_CELL_PX = 3

def get_grid_dimensions(cell_px: int) -> Tuple[int, int]:
    """
    Retorna (grid_cols, max_rows) otimizados para a largura disponível na folha A4 (2160 x 2520 px)
    mantendo a grade centrada e dentro dos marcadores fiduciais.
    """
    cell = max(1, int(cell_px))
    cols = (2160 // cell // 4) * 4
    rows = (2520 // cell // 4) * 4
    return cols, rows

# Dimensões da fita de calibração
CALIBRATION_STRIP_HEIGHT_PX = 48
CALIBRATION_MARGIN_TOP_PX = 16

# Marcadores fiduciais
FIDUCIAL_SIZE_PX = 96
FIDUCIAL_MARGIN_PX = 90

# QR Code no cabeçalho (não sobrepõe o fiducial TL em 90..186)
QR_CODE_X_PX = 230
QR_CODE_Y_PX = 60
QR_CODE_SIZE_PX = 320
HEADER_MARGIN_TOP_PX = 60

# Instrução canônica para auto-reconstrução
UNIVERSAL_INSTRUCTION = (
    "Scan QR for params. Sample cell centers using 4 corner fiducials. "
    "Classify levels via bottom calibration strip. Deinterleave 2D array. "
    "Correct errors via Reed-Solomon ECC. Decompress zlib payload. Verify SHA-256."
)

@dataclass
class PageLayout:
    page_w: int = PAGE_WIDTH_A4
    page_h: int = PAGE_HEIGHT_A4
    grid_cols: int = DEFAULT_GRID_COLS
    grid_rows: int = DEFAULT_GRID_ROWS
    cell_px: int = DEFAULT_CELL_PX
    mode: str = MODE_GRAY4
    copies: int = DEFAULT_COPIES
    ecc_percent: int = DEFAULT_ECC_PERCENT
    show_text: bool = True

    def __post_init__(self):
        # Se cell_px for alterado e cols/rows ainda forem os padrões, recalcula automaticamente
        if self.cell_px != DEFAULT_CELL_PX and self.grid_cols == DEFAULT_GRID_COLS and self.grid_rows == DEFAULT_GRID_ROWS:
            self.grid_cols, self.grid_rows = get_grid_dimensions(self.cell_px)

    @property
    def grid_width_px(self) -> int:
        return self.grid_cols * self.cell_px

    @property
    def grid_height_px(self) -> int:
        return self.grid_rows * self.cell_px

    @property
    def grid_left(self) -> int:
        return (self.page_w - self.grid_width_px) // 2

    @property
    def grid_top(self) -> int:
        # Espaço reservado para o cabeçalho e QR Code
        return 520

    @property
    def total_cells(self) -> int:
        return self.grid_cols * self.grid_rows
