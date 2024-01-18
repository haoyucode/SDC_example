#* Purpose: create uni, bi, and trigrams from text data
#*
#* Date: 2022.08.10

# Setup ----

library(tidyverse)
library(data.table)
library(text2vec)
library(tm)

# Define ----

# Function to training tokens & generate n-grams
train_and_ngram <- 
  function(
    data,
    id,
    text_col,
    n_grams = 10,
    more_stopwords = NULL
  ) {
  
    # Define Data
    df <- data
    
    # Train tokens 
    train_tokens <- df[text_col] %>% 
      tolower %>% 
      text2vec::word_tokenizer()
    
    it_train <- text2vec::itoken(train_tokens,
                                 ids = df[id],
                                 progressbar = FALSE)
    
    # Conditional list of stop words
    if (is.null(more_stopwords)) {
      stop_words <- c(tm::stopwords("english"))
    } else if (!is.null(more_stopwords)) {
      stop_words <- c(tm::stopwords("english"),more_stopwords)
    }
    
    # Generate uni, bi, & trigrams
    gram1 <- text2vec::create_vocabulary(it_train, stopwords = stop_words, ngram=c(1L, 1L)) %>% 
      text2vec::prune_vocabulary(term_count_min=10) %>%
      arrange(desc(term_count)) %>% 
      rename(gram1=term,
             term_count1=term_count,
             doc_count1=doc_count)
    
    gram2 <- text2vec::create_vocabulary(it_train, stopwords = stop_words, ngram=c(2L, 2L)) %>% 
      text2vec::prune_vocabulary(term_count_min=5) %>%
      arrange(desc(term_count)) %>% 
      rename(gram2=term,
             term_count2=term_count,
             doc_count2=doc_count)
    
    gram3 <- text2vec::create_vocabulary(it_train, stopwords = stop_words, ngram=c(3L, 3L)) %>% 
      text2vec::prune_vocabulary(term_count_min=5) %>%
      arrange(desc(term_count)) %>% 
      rename(gram3=term,
             term_count3=term_count,
             doc_count3=doc_count)
    
    # Combine top n for each gram
    full <- bind_cols(
      head(gram1,n=n_grams),
      head(gram2,n=n_grams),
      head(gram3,n=n_grams)
    ) %>%
      select(gram1,gram2,gram3)
  
    # Export
    return(full)
  
}
