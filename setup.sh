mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# Install necessary system dependencies
apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev
