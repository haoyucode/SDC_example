#* Purpose: apply VADER sentiment scores
#*
#* Date: 2022.08.10

# Setup ----

library(tidyverse)
library(vader)

# Define ----

apply_vader <- 
  function(
    data,
    text_col
  ) {
    
    # Define data
    df <- data
    
    # Initial message
    print("Applying sentiment...", quote = FALSE)
    
    # Apply sentiment
    df <- df %>% 
      mutate(
        sentiment=vader::vader_df(text = .[text_col])
      )
    
    # Extraction message
    print("Extracting positive, negative, neutral, & compound scores...", quote = FALSE)
    
    # Extract pos, neg, neu, & compound scores
    final <- df %>% 
      mutate(
        vader.pos=.$sentiment$pos,
        vader.neg=.$sentiment$neg,
        vader.neu=.$sentiment$neu,
        vader.compound=.$sentiment$compound
      ) %>% 
      # Remove additional nested df
      select(-sentiment)
    
    # Export
    return(final)
    
}
