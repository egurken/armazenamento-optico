"""
Interface de Linha de Comando (CLI) para o DataSheet.
"""

import argparse
import os
import sys
from datasheet.config import (
    SUPPORTED_MODES,
    DEFAULT_COPIES,
    MIN_COPIES,
    MAX_COPIES,
    MODE_GRAY4
)
from datasheet.encoder import encode_file
from datasheet.decoder import decode_files, decode_page_image
import cv2


def main():
    parser = argparse.ArgumentParser(
        prog="datasheet",
        description="DataSheet - Armazenamento Óptico em Papel (4 tons de cinza, 8 cores, 16 cores e 32 cores)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando encode
    p_encode = subparsers.add_parser("encode", help="Codifica um arquivo em uma ou mais folhas A4 (PDF / PNG)")
    p_encode.add_argument("input", help="Arquivo a ser codificado")
    p_encode.add_argument("-o", "--output", help="Caminho do arquivo PDF consolidado de saída (ex: documento.pdf)")
    p_encode.add_argument("--png-dir", help="Diretório opcional para salvar as imagens PNG individuais")
    p_encode.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=MODE_GRAY4,
        help="Modo de modulação: gray4 (padrão P&B), color8 (8 cores RGB/CMY), color16 (16 cores) ou color32 (32 cores)"
    )
    p_encode.add_argument(
        "--copies",
        type=int,
        default=DEFAULT_COPIES,
        help=f"Número de cópias para redundância legada ({MIN_COPIES} a {MAX_COPIES}, padrão {DEFAULT_COPIES})"
    )
    p_encode.add_argument(
        "--ecc",
        type=int,
        default=20,
        help="Porcentagem de redundância matemática Reed-Solomon (5 a 50%%, padrão 20%%)"
    )
    p_encode.add_argument(
        "--cell-size",
        type=int,
        default=3,
        choices=[2, 3, 4, 5, 6],
        help="Tamanho da célula em pixels a 300 DPI (2=alta densidade, 3=padrão, 4=alta robustez, 5=fácil leitura, 6=máxima nitidez)"
    )
    p_encode.add_argument(
        "--cell-gap",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Espaçamento branco anti-sangramento entre células (0=sem borda, 1=1px recomendado para jato de tinta, 2=2px isolamento máximo)"
    )
    p_encode.add_argument(
        "--no-text",
        action="store_true",
        help="Oculta o texto legível e instrução no cabeçalho (mantém apenas o QR code obrigatório)"
    )

    # Subcomando decode
    p_decode = subparsers.add_parser("decode", help="Decodifica uma ou mais folhas escaneadas e recupera o arquivo")
    p_decode.add_argument("inputs", nargs="+", help="Arquivos de imagem (PNG, JPG) ou PDF escaneado")
    p_decode.add_argument("-o", "--output", required=True, help="Caminho do arquivo de saída recuperado")

    # Subcomando info
    p_info = subparsers.add_parser("info", help="Inspeciona os metadados de uma folha a partir do QR Code")
    p_info.add_argument("input", help="Arquivo de imagem ou PDF da folha")

    args = parser.parse_args()

    if args.command == "encode":
        if not (MIN_COPIES <= args.copies <= MAX_COPIES):
            print(f"[Erro] O número de cópias deve estar entre {MIN_COPIES} e {MAX_COPIES}.", file=sys.stderr)
            sys.exit(1)

        pdf_out = args.output
        if not pdf_out and not args.png_dir:
            base = os.path.splitext(os.path.basename(args.input))[0]
            pdf_out = f"{base}_datasheet.pdf"

        print(f"Codificando arquivo: {args.input}")
        print(f"Modo: {args.mode} | Reed-Solomon ECC: {args.ecc}% | Célula: {args.cell_size}px (Gap: {args.cell_gap}px) | Formato: A4")
        pages = encode_file(
            input_path=args.input,
            output_pdf_path=pdf_out,
            output_png_dir=args.png_dir,
            mode=args.mode,
            copies=args.copies,
            ecc_percent=args.ecc,
            cell_size=args.cell_size,
            show_text=not args.no_text,
            cell_gap=args.cell_gap
        )
        print(f"[Sucesso] {len(pages)} página(s) gerada(s).")
        if pdf_out:
            print(f"PDF consolidado: {pdf_out}")
        if args.png_dir:
            print(f"PNGs salvos em: {args.png_dir}")

    elif args.command == "decode":
        print(f"Iniciando decodificação de {len(args.inputs)} arquivo(s)...")
        try:
            raw_data, orig_filename = decode_files(args.inputs, output_path=args.output)
            print(f"Arquivo recuperado com sucesso: {args.output} ({len(raw_data)} bytes)")
        except Exception as e:
            print(f"[Erro na decodificação] {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "info":
        from datasheet.decoder import extract_images_from_pdf
        from datasheet.metadata import read_qr_from_image

        if args.input.lower().endswith(".pdf"):
            images = extract_images_from_pdf(args.input)
            img = images[0] if images else None
        else:
            img = cv2.imread(args.input)

        if img is None:
            print(f"[Erro] Não foi possível carregar: {args.input}", file=sys.stderr)
            sys.exit(1)

        meta = read_qr_from_image(img)
        if not meta:
            print("[Erro] QR Code de controle não encontrado nesta imagem.", file=sys.stderr)
            sys.exit(1)

        print("\n=== METADADOS DA FOLHA ===")
        print(f"ID do Arquivo (10 dígitos): {meta.file_id}")
        print(f"Folha:                      {meta.page} de {meta.pages}")
        print(f"Nome Original:              {meta.file_name}")
        print(f"Modo de Cor:                {meta.mode}")
        print(f"Fator de Redundância:       {meta.copies} cópias")
        print(f"Tamanho Bruto:              {meta.raw_size} bytes")
        print(f"Tamanho Comprimido:         {meta.comp_size} bytes")
        print(f"Grade:                      {meta.grid_cols}x{meta.grid_rows} células ({meta.cell_px}px/célula, gap {getattr(meta, 'cell_gap', 0)}px)")
        print(f"SHA-256 Global:             {meta.full_sha256}")
        print(f"SHA-256 do Bloco:           {meta.chunk_sha256}")
        print("==========================\n")


if __name__ == "__main__":
    main()
