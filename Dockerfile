FROM animcogn/face_recognition:gpu

# Install and configure poetry
RUN pip install --upgrade pip
RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /app
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

# Copy source files
COPY src/ ./
