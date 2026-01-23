FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY app/requirements_frontend.txt /app/requirements_frontend.txt
COPY app/frontend.py /app/frontend.py

RUN  uv pip install --system --no-cache -r requirements_frontend.txt

EXPOSE 8501

#ENV PYTHONPATH="${PYTHONPATH}:/root/.local/lib/python3.12/site-packages"

ENTRYPOINT python -m streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
#ENTRYPOINT ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#ENTRYPOINT uv run streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
#ENTRYPOINT ["uv", "run", "streamlit", "run", "frontend.py", "--server.port", "${PORT}", "--server.address=0.0.0.0"]