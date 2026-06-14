FROM python:3.11-slim

WORKDIR /code

# Force Python to stream stdout/stderr logs directly to the console without buffering
ENV PYTHONUNBUFFERED=1

# Copy all uploaded flat binaries and scripts straight into /code
COPY . /code/

# Install the dependencies cleanly
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Expose the standard Hugging Face proxy routing port
EXPOSE 7860

# Execute Uvicorn pointing directly to main.py in the flat root
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]