# NOTE: 
# This is R code and this code was run in Rstudio 
# This code is used to perform ANOVA analysis on the ART data
# Load necessary libraries
install.packages("ARTool")
library(ARTool)
library(dplyr)

# Load the dataset
file_path <- "/Users/noahvanrijn/python-repos/master/Thesis/results/QA/Final_QA.csv" # Change this to the path of your CSV file
data <- read.csv(file_path)

# Convert categorical variables to factors
data$Chatbot <- as.factor(data$Chatbot)
data$Psychologist_ID <- as.factor(data$Psychologist_ID)

# Function to perform ART ANOVA for a given criterion
perform_art_anova <- function(data, criterion) {
  formula <- as.formula(paste(criterion, "~ Chatbot * Psychologist_ID"))
  art_model <- art(formula, data = data)
  anova_results <- anova(art_model)
  print(paste("ANOVA Results for", criterion))
  print(anova_results)
}

# Perform ART ANOVA for each criterion
criteria <- c("Informativeness", "Guidance", "Empathy", "Relevance", "Understanding")

for (criterion in criteria) {
  perform_art_anova(data, criterion)
}
