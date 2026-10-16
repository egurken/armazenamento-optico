"""
datasheet/printer.py - Utilitários de impressão nativa para Windows
Permite listar impressoras disponíveis e enviar páginas diretamente para o Spooler do Windows.
"""

import os
import sys
import subprocess
import json
import logging

logger = logging.getLogger(__name__)

def get_windows_printers():
    """Retorna uma lista de dicionários com as impressoras instaladas no Windows."""
    if sys.platform != "win32":
        return []

    ps_script = """
    $printers = Get-CimInstance Win32_Printer | Select-Object Name, Default, DriverName
    $result = @()
    foreach ($p in $printers) {
        $result += @{
            name = $p.Name
            is_default = [bool]$p.Default
            driver = $p.DriverName
        }
    }
    $result | ConvertTo-Json -Compress
    """
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=8
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout.strip())
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
    except Exception as e:
        logger.warning(f"Erro ao listar impressoras do Windows: {e}")

    return []


def print_images_windows(image_paths, printer_name=None):
    """
    Envia uma lista de imagens (PNGs em resolução A4 300 DPI)
    diretamente para a impressora do Windows via .NET System.Drawing.Printing.PrintDocument,
    utilizando interpolação NearestNeighbor (vizinho mais próximo) para evitar borramento
    e sobreposição de cores entre células adjacentes.
    """
    if sys.platform != "win32":
        return False, "Impressão direta só é suportada no Windows."

    valid_paths = [os.path.abspath(p) for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        return False, "Nenhum arquivo de imagem encontrado para imprimir."

    # Prepara caminhos formatados para o PowerShell
    escaped_paths = ", ".join([f"'{p.replace('\'', '\'\'')}'" for p in valid_paths])
    printer_arg = f"'{printer_name.replace('\'', '\'\'')}'" if printer_name else "$null"

    ps_script = f"""
    $ErrorActionPreference = 'Stop'
    Add-Type -AssemblyName System.Drawing

    $images = @({escaped_paths})
    $targetPrinter = {printer_arg}

    $doc = New-Object System.Drawing.Printing.PrintDocument
    if ($targetPrinter) {{
        $doc.PrinterSettings.PrinterName = $targetPrinter
    }}

    if (-not $doc.PrinterSettings.IsValid) {{
        Write-Error "Impressora não encontrada ou inválida: $($doc.PrinterSettings.PrinterName)"
        exit 1
    }}

    # Configuração de alta fidelidade para papel A4 sem margens adicionais de escalonamento
    $doc.DefaultPageSettings.Margins = New-Object System.Drawing.Printing.Margins(0, 0, 0, 0)
    $doc.DefaultPageSettings.Color = $true
    $doc.OriginAtMargins = $true

    foreach ($p in $doc.PrinterSettings.PaperSizes) {{
        if ($p.PaperName -like "*A4*") {{
            $doc.DefaultPageSettings.PaperSize = $p
            break
        }}
    }}

    # Seleciona a resolução máxima (High) suportada pelo driver para maior nitidez de ponto
    foreach ($res in $doc.PrinterSettings.PrinterResolutions) {{
        if ($res.Kind -eq [System.Drawing.Printing.PrinterResolutionKind]::High) {{
            $doc.DefaultPageSettings.PrinterResolution = $res
            break
        }}
    }}

    $script:pageIdx = 0

    $doc.add_PrintPage({{
        param($sender, $e)
        if ($script:pageIdx -lt $images.Count) {{
            $imgFile = $images[$script:pageIdx]
            $img = [System.Drawing.Image]::FromFile($imgFile)

            $rect = $e.MarginBounds
            if ($rect.Width -le 0 -or $rect.Height -le 0) {{
                $rect = $e.PageBounds
            }}

            # CRÍTICO: NearestNeighbor + PixelOffsetMode::Half evita interpolação bicúbica
            # que mistura as cores de células vizinhas na hora de imprimir
            $e.Graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
            $e.Graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half
            $e.Graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::None
            $e.Graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::AssumeLinear
            $e.Graphics.DrawImage($img, $rect)

            $img.Dispose()
            $script:pageIdx++

            $e.HasMorePages = ($script:pageIdx -lt $images.Count)
        }} else {{
            $e.HasMorePages = $false
        }}
    }})

    $doc.Print()
    Write-Output "SUCESSO:$($doc.PrinterSettings.PrinterName)"
    """

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        if res.returncode == 0 and "SUCESSO:" in res.stdout:
            actual_printer = res.stdout.split("SUCESSO:")[1].strip()
            return True, f"{len(valid_paths)} folha(s) enviada(s) com sucesso para a fila de impressão da '{actual_printer}'!"
        else:
            err_msg = res.stderr.strip() or res.stdout.strip()
            return False, f"Falha ao imprimir via Windows: {err_msg}"
    except Exception as e:
        return False, f"Erro ao executar comando de impressão: {e}"
