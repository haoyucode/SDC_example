#* Purpose: compare similarity scores by group for time frame of interest.
#*
#* Date: 2022.08.16

# Setup ----

library(tidyverse)
library(text2vec)
library(lubridate)

# Define ----

# Function definition
similarity <- 
  function(
    data,
    comparison_group,
    all_group,
    time_col#,
    #time_interval = "month"
  ) {
    
    # Define data
    df <-  data
    
    # Rename grouping column to generic: comparison or all groups
    vars_rename <- c(comparison_group,all_group)

    df <- rename_at(
      df,
      vars(all_of(vars_rename)),
      ~c("comparison_group","all_group")
    )
    
    # Rename time to generic to work with text2vec::psim2()
    names(df)[names(df) == time_col ] <- 'time'

    # Tokenize by group
    tokens <- list(
      "comparison_group" = {
        df$comparison_group %>%
          word_tokenizer() %>%
          itoken(ids=df$time,
                 progressbar = FALSE)
      },
      "all_group" = {
        df$all_group %>%
          word_tokenizer() %>%
          itoken(ids=df$time,
                 progressbar = FALSE)
      }
    )

    # Prune vocab of all group
    pruned_vocab <- create_vocabulary(tokens[["all_group"]]) %>%
      prune_vocabulary(
        doc_proportion_max = 0.9,
        term_count_min = 5
      )

    # Vectorize pruned vocab
    pruned_vectorizer <- vocab_vectorizer(pruned_vocab)

    # Document-term matrices by group
    dtm_comparison <- create_dtm(tokens[["comparison_group"]], pruned_vectorizer)
    dtm_all <- create_dtm(tokens[["all_group"]], pruned_vectorizer)

    # TF-IDF Transformation
    tfidf <- TfIdf$new()

    dtm_tf_comparison <- fit_transform(dtm_comparison, tfidf)
    dtm_tf_all <- fit_transform(dtm_all, tfidf)

    # Compute similarity (Jaccard & cosine)
    scores <- list(
      "jac"={
        psim2(
          dtm_tf_comparison,
          dtm_tf_all,
          method="jaccard",
          norm="none"
        )
      },
      "cos"={
        psim2(
          dtm_tf_comparison,
          dtm_tf_all,
          method="cosine",
          norm="l2"
        )
      }
    )
    
    # Scores to dataframes
    scores_df <- list(
      "jaccard"={
        data.frame(
          id=names(scores[["jac"]]),
          jaccard_score=scores[["jac"]],
          row.names = NULL
        )
      },
      "cosine"={
        data.frame(
          id=names(scores[["cos"]]),
          cosine_score=scores[["cos"]],
          row.names = NULL
        )
      }
    )
    
    # Scores for export
    export_scores <- 
      scores_df[["jaccard"]] %>% 
      left_join(scores_df[["cosine"]], by = "id")

    # Export
    return(export_scores)
    
  }
