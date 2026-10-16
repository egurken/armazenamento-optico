"""
DataSheet Web UI - Servidor Flask local com interface gráfica amigável.
"""

import os
import uuid
import shutil
import tempfile
from typing import Optional
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

from datasheet.config import (
    SUPPORTED_MODES,
    DEFAULT_COPIES,
    DEFAULT_ECC_PERCENT,
    DEFAULT_CELL_PX,
    DEFAULT_CELL_GAP,
    MODE_GRAY4,
    MODE_COLOR8,
    MODE_COLOR16,
    MODE_COLOR32,
    PageLayout
)
from datasheet.encoder import encode_file
from datasheet.decoder import decode_files, extract_images_from_pdf, decode_page_image, load_image_any_path
from datasheet.metadata import read_qr_from_image
from datasheet.chunker import calculate_page_capacity
from datasheet.printer import get_windows_printers, print_images_windows

app = Flask(__name__)

# Diretório base para arquivos temporários de sessão
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/encode", methods=["POST"])
def api_encode():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Nome de arquivo inválido."}), 400

    mode = request.form.get("mode", MODE_COLOR8)
    if mode not in SUPPORTED_MODES:
        mode = MODE_COLOR8

    try:
        ecc_percent = int(request.form.get("ecc_percent", DEFAULT_ECC_PERCENT))
        ecc_percent = max(5, min(50, ecc_percent))
    except ValueError:
        ecc_percent = DEFAULT_ECC_PERCENT

    try:
        cell_size = int(request.form.get("cell_size", DEFAULT_CELL_PX))
        cell_size = max(2, min(6, cell_size))
    except ValueError:
        cell_size = DEFAULT_CELL_PX

    try:
        cell_gap = int(request.form.get("cell_gap", DEFAULT_CELL_GAP))
        cell_gap = max(0, min(2, cell_gap))
    except ValueError:
        cell_gap = DEFAULT_CELL_GAP

    try:
        copies = int(request.form.get("copies", DEFAULT_COPIES))
        copies = max(3, min(6, copies))
    except ValueError:
        copies = DEFAULT_COPIES

    show_text = request.form.get("show_text", "true").lower() == "true"

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, uploaded_file.filename)
    uploaded_file.save(input_path)

    base_name = os.path.splitext(uploaded_file.filename)[0]
    safe_base = "".join(c for c in base_name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_base:
        safe_base = "documento"
    pdf_filename = f"{safe_base}.pdf"
    pdf_output_path = os.path.join(session_dir, pdf_filename)
    png_dir = os.path.join(session_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)

    try:
        pages = encode_file(
            input_path=input_path,
            output_pdf_path=pdf_output_path,
            output_png_dir=png_dir,
            mode=mode,
            copies=copies,
            ecc_percent=ecc_percent,
            cell_size=cell_size,
            show_text=show_text,
            cell_gap=cell_gap
        )

        png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
        preview_urls = [f"/api/preview/{session_id}/{f}" for f in png_files]

        # Inspeciona metadados da primeira página de forma segura com acentuação
        first_img = load_image_any_path(os.path.join(png_dir, png_files[0]))
        meta = read_qr_from_image(first_img) if first_img is not None else None

        layout = PageLayout(mode=mode, copies=copies, ecc_percent=ecc_percent, cell_px=cell_size, cell_gap=cell_gap)
        _, useful_chunk_bytes = calculate_page_capacity(layout)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "file_name": uploaded_file.filename,
            "file_id": meta.file_id if meta else "N/A",
            "total_pages": len(pages),
            "mode": mode,
            "copies": copies,
            "ecc_percent": ecc_percent,
            "cell_size": cell_size,
            "cell_gap": cell_gap,
            "raw_size": meta.raw_size if meta else os.path.getsize(input_path),
            "comp_size": meta.comp_size if meta else 0,
            "page_capacity": useful_chunk_bytes,
            "full_sha256": meta.full_sha256 if meta else "",
            "pdf_url": f"/api/download_pdf/{session_id}",
            "previews": preview_urls
        })
    except Exception as e:
        return jsonify({"error": f"Erro durante a codificação: {str(e)}"}), 500


def find_session_pdf(session_dir: str) -> Optional[str]:
    """Localiza o arquivo PDF compilado na pasta da sessão."""
    for f in os.listdir(session_dir):
        if f.endswith(".pdf"):
            return os.path.join(session_dir, f)
    return None


@app.route("/api/printers")
def api_printers():
    """Retorna as impressoras disponíveis no Windows para seleção na interface."""
    printers = get_windows_printers()
    return jsonify({"printers": printers})


@app.route("/print/<session_id>")
def view_print_session(session_id):
    """Renderiza todas as páginas da sessão prontas para impressão direta no navegador."""
    session_dir = os.path.join(STORAGE_DIR, session_id)
    if not os.path.exists(session_dir):
        return "Sessão não encontrada", 404
    png_dir = os.path.join(session_dir, "pngs")
    if not os.path.exists(png_dir):
        return "Páginas não encontradas", 404
    pngs = sorted([f for f in os.listdir(png_dir) if f.lower().endswith(".png")])
    page_urls = [f"/api/preview/{session_id}/{p}" for p in pngs]
    return render_template("print_view.html", pages=page_urls, session_id=session_id)


@app.route("/api/preview/<session_id>/<filename>")
def api_preview(session_id, filename):
    png_dir = os.path.join(STORAGE_DIR, session_id, "pngs")
    return send_from_directory(png_dir, filename)


@app.route("/api/download_pdf/<session_id>")
def api_download_pdf(session_id):
    session_dir = os.path.join(STORAGE_DIR, session_id)
    if not os.path.exists(session_dir):
        return jsonify({"error": "Sessão não encontrada."}), 404
    pdf_path = find_session_pdf(session_dir)
    if pdf_path and os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))
    return jsonify({"error": "PDF não encontrado."}), 404


@app.route("/api/print_windows/<session_id>", methods=["POST"])
def api_print_windows(session_id):
    """
    Envia as folhas em alta resolução diretamente para o Spooler do Windows (.NET PrintDocument),
    permitindo escolher a impressora desejada ou usando a padrão.
    """
    session_dir = os.path.join(STORAGE_DIR, session_id)
    if not os.path.exists(session_dir):
        return jsonify({"error": "Sessão não encontrada."}), 404

    png_dir = os.path.join(session_dir, "pngs")
    if not os.path.exists(png_dir):
        return jsonify({"error": "Imagens da sessão não encontradas."}), 404

    png_files = sorted([os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.lower().endswith(".png")])
    if not png_files:
        return jsonify({"error": "Nenhuma folha gerada encontrada para impressão."}), 404

    # Extrai o nome da impressora se passado pelo usuário
    data = request.get_json(silent=True) or {}
    printer_name = data.get("printer_name") or request.args.get("printer_name")
    if printer_name:
        printer_name = printer_name.strip()

    ok, msg = print_images_windows(png_files, printer_name=printer_name)
    if ok:
        return jsonify({
            "success": True,
            "message": msg
        })
    else:
        # Se a fila direta falhar, tenta como contingência abrir o PDF no leitor
        pdf_path = find_session_pdf(session_dir)
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.startfile(pdf_path)
            except Exception:
                pass
        return jsonify({"error": msg}), 500


@app.route("/api/decode", methods=["POST"])
def api_decode():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files or len(uploaded_files) == 0:
        return jsonify({"error": "Nenhum arquivo enviado para decodificação."}), 400

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_paths = []
    for f in uploaded_files:
        if f.filename:
            save_path = os.path.join(session_dir, f.filename)
            f.save(save_path)
            input_paths.append(save_path)

    try:
        # Extrai imagens de PDFs ou carrega arquivos de imagem com suporte a acentos
        all_imgs = []
        for p in input_paths:
            if p.lower().endswith(".pdf"):
                all_imgs.extend(extract_images_from_pdf(p))
            else:
                img = load_image_any_path(p)
                if img is not None:
                    all_imgs.append(img)

        if not all_imgs:
            return jsonify({"error": "Nenhuma imagem válida encontrada nos arquivos enviados."}), 400

        from datasheet.chunker import PageAssembler
        assembler = PageAssembler()
        detected_pages_info = []

        for i, img in enumerate(all_imgs):
            try:
                meta, chunk_bytes = decode_page_image(img)
                ok = assembler.add_page(meta, chunk_bytes)
                detected_pages_info.append({
                    "page": meta.page,
                    "total_pages": meta.pages,
                    "file_id": meta.file_id,
                    "valid": ok
                })
            except Exception as pe:
                detected_pages_info.append({
                    "page": i + 1,
                    "error": str(pe),
                    "valid": False
                })

        if not assembler.is_complete():
            if len(assembler.pages_data) == 0:
                first_err = detected_pages_info[0].get("error", "") if detected_pages_info else ""
                hint = ""
                if "danos físicos" in first_err or "Reed-Solomon" in first_err or "correção de erros" in first_err:
                    hint = " Dica: para impressoras jato de tinta em papel comum, ative o Isolamento Anti-Sangramento (1px ou 2px de borda branca) e utilize ponto de 4px a 6px no modo 8 Cores ou 4 Tons de Cinza."
                elif "QR Code" in first_err:
                    hint = " Dica: certifique-se de que a folha inteira e o QR Code do cabeçalho estejam visíveis na imagem."
                return jsonify({
                    "success": False,
                    "error": f"Não foi possível recuperar os dados da folha.{f' Motivo: {first_err}.' if first_err else ''}{hint}",
                    "detected": detected_pages_info
                }), 400
            missing = assembler.get_missing_pages()
            pages_read = sorted(list(assembler.pages_data.keys()))
            return jsonify({
                "success": False,
                "incomplete": True,
                "received_pages": len(assembler.pages_data),
                "total_pages": assembler.total_pages or len(all_imgs),
                "missing_pages": missing,
                "detected": detected_pages_info,
                "error": f"Documento multi-páginas incompleto! Foram lidas as páginas {pages_read}, mas ainda faltam as páginas {missing} para reconstruir o arquivo."
            }), 400

        raw_data, orig_filename = assembler.assemble()
        recovered_path = os.path.join(session_dir, orig_filename)
        with open(recovered_path, "wb") as rf:
            rf.write(raw_data)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "file_name": orig_filename,
            "file_id": assembler.file_id,
            "raw_size": len(raw_data),
            "total_pages": assembler.total_pages,
            "sha256": assembler.full_sha256,
            "download_url": f"/api/download_recovered/{session_id}/{orig_filename}",
            "detected": detected_pages_info
        })
    except Exception as e:
        return jsonify({"error": f"Erro na decodificação: {str(e)}"}), 500


@app.route("/api/download_recovered/<session_id>/<filename>")
def api_download_recovered(session_id, filename):
    rec_path = os.path.join(STORAGE_DIR, session_id, filename)
    if os.path.exists(rec_path):
        return send_file(rec_path, as_attachment=True, download_name=filename)
    return jsonify({"error": "Arquivo recuperado não encontrado."}), 404


if __name__ == "__main__":
    import webbrowser
    print("\n" + "="*60)
    print("  DataSheet Web UI Iniciado com Sucesso!")
    print("  Acesse no seu navegador: http://127.0.0.1:5000")
    print("="*60 + "\n")
    # Abre o navegador automaticamente
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
