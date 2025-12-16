#!/bin/bash
# Book Build Script - Converts Markdown chapters to various formats

BOOK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$BOOK_DIR/output"

mkdir -p "$OUTPUT_DIR"

echo "Building book from: $BOOK_DIR"
echo "Output to: $OUTPUT_DIR"

# Check for pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Pandoc not found. Install with: brew install pandoc"
    echo "Or download from: https://pandoc.org/installing.html"
    exit 1
fi

# Build single chapter (Chapter 2)
echo ""
echo "=== Building Chapter 2 ==="

CHAPTER_DIR="$BOOK_DIR/chapters/02-ai-driven-optimization"
CHAPTER="$CHAPTER_DIR/content/chapter-02.md"
FIGURES="$CHAPTER_DIR/figures"

# Resource path for images (so pandoc can find ../figures/ references)
RESOURCE_PATH="$CHAPTER_DIR/content:$CHAPTER_DIR"

# 1. Export to PDF (requires LaTeX)
echo "Generating PDF..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-02.pdf" \
    --resource-path="$RESOURCE_PATH" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc \
    --highlight-style=tango \
    2>/dev/null && echo "  ✓ chapter-02.pdf" || echo "  ✗ PDF failed (install MacTeX for PDF support)"

# 2. Export to Word (DOCX)
echo "Generating Word document..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-02.docx" \
    --resource-path="$RESOURCE_PATH" \
    --toc \
    --highlight-style=tango \
    && echo "  ✓ chapter-02.docx"

# 3. Export to HTML (with embedded images)
echo "Generating HTML..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-02.html" \
    --resource-path="$RESOURCE_PATH" \
    --standalone \
    --self-contained \
    --toc \
    --highlight-style=tango \
    -c "https://cdn.jsdelivr.net/npm/github-markdown-css@5.2.0/github-markdown.min.css" \
    && echo "  ✓ chapter-02.html"

# 4. Export to EPUB (ebook)
echo "Generating EPUB..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-02.epub" \
    --resource-path="$RESOURCE_PATH" \
    --toc \
    && echo "  ✓ chapter-02.epub"

# Build Chapter 3
echo ""
echo "=== Building Chapter 3 ==="

CHAPTER_DIR="$BOOK_DIR/chapters/03-comparative-framework"
CHAPTER="$CHAPTER_DIR/content/chapter-03.md"
RESOURCE_PATH="$CHAPTER_DIR/content:$CHAPTER_DIR"

# 1. Export to PDF (requires LaTeX)
echo "Generating PDF..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-03.pdf" \
    --resource-path="$RESOURCE_PATH" \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc \
    --highlight-style=tango \
    2>/dev/null && echo "  ✓ chapter-03.pdf" || echo "  ✗ PDF failed (install MacTeX for PDF support)"

# 2. Export to Word (DOCX)
echo "Generating Word document..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-03.docx" \
    --resource-path="$RESOURCE_PATH" \
    --toc \
    --highlight-style=tango \
    && echo "  ✓ chapter-03.docx"

# 3. Export to HTML (with embedded images)
echo "Generating HTML..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-03.html" \
    --resource-path="$RESOURCE_PATH" \
    --standalone \
    --self-contained \
    --toc \
    --highlight-style=tango \
    -c "https://cdn.jsdelivr.net/npm/github-markdown-css@5.2.0/github-markdown.min.css" \
    && echo "  ✓ chapter-03.html"

# 4. Export to EPUB (ebook)
echo "Generating EPUB..."
pandoc "$CHAPTER" \
    -o "$OUTPUT_DIR/chapter-03.epub" \
    --resource-path="$RESOURCE_PATH" \
    --toc \
    && echo "  ✓ chapter-03.epub"

echo ""
echo "=== Build Complete ==="
echo "Files in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
