# Base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional package
# RUN pip install --no-cache-dir langchain-community

# Copy the application code
COPY . .

# Command to run the application
CMD ["streamlit", "run", "chatpdf.py"]