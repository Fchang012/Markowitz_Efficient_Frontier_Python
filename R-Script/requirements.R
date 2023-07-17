# List of packages
packages <- c("quantmod")

# Function to check and install packages
check_and_install <- function(pkg){
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Apply the function to each package
sapply(packages, check_and_install)

# Load the libraries
lapply(packages, library, character.only = TRUE)
