# Use rocker-versioned2
# https://github.com/rocker-org/rocker-versioned2
FROM rocker/shiny-verse

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install R packages
RUN install2.r --error \
    xts \
    zoo \
    TTR \
    methods \
    curl \
    quantmod

# Make port 80 available to the world outside this container
EXPOSE 80

# Use an entrypoint script to run Rscript
ENTRYPOINT ["Rscript"]
