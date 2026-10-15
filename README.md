# DataSheet - Armazenamento Óptico de Dados em Papel

> ⚠️ **AVISO: PROJETO EM ESTÁGIO BETA / NÃO ESTÁ FUNCIONANDO TOTALMENTE AINDA**
>
> Este repositório é uma **prova de conceito experimental** para arquivamento e recuperação física de dados digitais em folhas de papel A4 comum.
>
> **O sistema AINDA NÃO está funcionando de forma estável ou confiável para uso prático/produção.** 
> Atualmente, a leitura óptica e a decodificação enfrentam desafios complexos de calibração espectral e alinhamento geométrico:
> - **Fotos de smartphone:** costumam falhar com frequência devido à subamostragem de croma (compressão JPEG 4:2:0), aberração cromática de lentes, iluminação não uniforme, sombras e desfoque em células microscópicas.
> - **Scanners de mesa (300 DPI):** têm maior taxa de sucesso em modos de baixa densidade (ex: 8 cores com células de 5px ou tons de cinza), mas modos de alta densidade (16 e 32 cores) ainda sofrem com sangramento físico de tinta (*ink bleed*) entre células adjacentes.
>
> O código está sendo publicado para fins de pesquisa, estudos em visão computacional, algoritmos de correção de erro (ECC) e colaboração com a comunidade. Contribuições, ideias e pull requests são muito bem-vindos!

---

## 💡 Sobre o Projeto

O **DataSheet** explora a conversão de arquivos digitais (documentos, chaves criptográficas, mídias compactadas) em matrizes de pontos bidimensionais impressos em papel A4 convencional, com recuperação de dados baseada em visão computacional e códigos corretores de erro.

---

## 🚀 Funcionalidades e Modos (Em Desenvolvimento)

- **4 Modos de Modulação Espectral:**
  - `gray4`: 4 tons de cinza PAM-4 (2 bits/célula) — Alta compatibilidade com impressoras laser P&B.
  - `color8`: 8 cores puras do cubo RGB/CMY (3 bits/célula) — Modo balanceado para impressoras coloridas.
  - `color16`: 16 cores calibradas em $L^*a^*b^*$ (4 bits/célula) — Alta densidade de dados.
  - `color32`: 32 cores calibradas (5 bits/célula) — Máxima densidade teórica por folha.
- **Correção de Erros Reed-Solomon + Entrelaçamento 2D:**
  - Suporte a seletor de redundância configurável (5% a 50%, padrão: 20%).
  - Embaralhamento helicoidal/matricial dos blocos de paridade por toda a folha para tolerar manchas, rasgos físicos e oclusão parcial.
  - Modo legado com repetição de 3 a 6 cópias e votação por maioria bit a bit.
- **QR Code de Auto-Configuração Obrigatório:**
  - Cabeçalho legível com metadados estruturados (identificador do arquivo, SHA-256 original, modo de cor, tamanho da célula, blocos).
- **Fita Espectral de Calibração:**
  - Amostras de referência no rodapé de cada folha para calibrar balanço de branco e distorção espectral introduzida pelo scanner/câmera.
- **Interface Web Interativa:**
  - Desenvolvida em Flask com preview visual em tempo real, estimativa de KB por folha, seleção de parâmetros e decodificador drag-and-drop.

---

## 📦 Instalação

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/egurken/armazenamento-optico.git
cd armazenamento-optico
pip install -r requirements.txt
```

---

## 🖥️ Como Executar a Interface Web

No Windows, execute o arquivo de inicialização:

```cmd
iniciar.bat
```

Ou via terminal:

```bash
python app.py
```

Em seguida, abra o navegador em: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠️ Como Usar (Linha de Comando - CLI)

### 1. Codificar um Arquivo para Impressão

```bash
# Modo 8 cores com células de 5px e 20% de Reed-Solomon
python -m datasheet.cli encode backup.zip -o backup_datasheet.pdf --mode color8 --cell-size 5 --ecc 20

# Modo tons de cinza (gray4)
python -m datasheet.cli encode chaves.tar -o chaves.pdf --mode gray4 --copies 5
```

### 2. Inspecionar uma Folha Gerada

```bash
python -m datasheet.cli info backup_datasheet.pdf
# ou
python -m datasheet.cli info pagina_001.png
```

### 3. Decodificar e Tentar Recuperar Arquivo

```bash
python -m datasheet.cli decode scan_folha.jpg -o arquivo_recuperado.bin
```

---

## 🧪 Testes Automatizados

O projeto inclui suíte de testes de validação unitária e roundtrip:

```bash
python -m unittest discover tests
```

---

## 🤝 Contribuições

Este projeto está em fase embrionária e precisa de ajuda especialmente nas seguintes frentes:
1. Algoritmos de deconvolução óptica e anti-bleeding para células pequenas (<= 4px).
2. Retificação de perspectiva não-linear para fotos de smartphone com distorção de lente (barrel distortion).
3. Classificação de cores sob diferentes fontes de luz (temperaturas de cor variáveis).

Pull requests e sugestões são bem-vindos!
