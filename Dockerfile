# Dockerfile

# Use the official PyMesh image from the Docker Hub
FROM pymesh/pymesh

# Set the working directory
WORKDIR /app

# Copy the Python script into the container
COPY simplify_mesh.py .

# Set the entry point to the Python script
ENTRYPOINT ["python", "./simplify_mesh.py"]
