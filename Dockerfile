# 1. Use the standard Python image (includes compilers)
FROM python:3.11

# 2. Install GNU Parallel (if your .sh script uses it to run python scripts)
RUN apt-get update && apt-get install -y \
    parallel \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# 3. Suppress GNU Parallel citation notice
RUN mkdir -p ~/.parallel && touch ~/.parallel/will-cite

# 4. Set the working directory
WORKDIR /app

# 5. Install cmdstanpy
RUN pip install --no-cache-dir cmdstanpy

# 6. Install CmdStan to its default location
# This is required so your Python scripts can run Stan models
RUN python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# 7. Install other Python dependencies from your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy your application code (including cross-validate.sh)
COPY . .

# 9. Make the shell script executable
RUN chmod +x cross-validate.sh

# 10. Final command: Run the shell script
# Since the script only runs python, we don't need to export any paths
CMD ["/bin/bash", "-c", ". cross-validate.sh"]