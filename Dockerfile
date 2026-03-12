## We will use the official Python 3.12 slim image as our base
FROM python:3.12-slim

## Define the working directory inside the container
WORKDIR /Setlu

## Copy the requirements file into the container
COPY requirements.txt .

## Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

## Copy the rest of the application code into the container
COPY . .

# Expose the port that the FastAPI app will run on
EXPOSE 8000

## Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]