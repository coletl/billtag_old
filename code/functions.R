# Functions to prepare ProPublica data ----
# This creates an object of class `bill` from PP's
# list (read from json) data on all Congressional bills

# Amendments are stored differently from bills. Only 3-4% of obs.
amend_to_bill <- function(data_json){
    
    bill_type         <- data_json$amendment_type
    bill_id           <- data_json$amendment_id
    committees        <- list()
    official_title    <- data_json$description
    short_title       <- sprintf("Amends bill %s", data_json$amends_bill$bill_id)
    cosponsors        <- NULL
    subjects          <- sprintf("Amends bill %s", data_json$amends_bill$bill_id)
    subjects_top_term <- sprintf("Amends bill %s", data_json$amends_bill$bill_id)
    summary           <- data.frame(text = data_json$purpose)
    history           <- data.frame(awaiting_signature = NA_integer_,
                                    enacted = NA_integer_,
                                    vetoed = NA_integer_)
    
    data_json$sponsor$state <-
        data_json$sponsor$district <-
        data_json$sponsor$title <- NA_character_
    
    out <- append(data_json,
                  dplyr::lst(bill_type,
                             bill_id,
                             committees,
                             official_title,
                             short_title,
                             cosponsors,
                             subjects,
                             subjects_top_term,
                             summary,
                             history))
    return(out)
}

# Convert from sponsor list and cosponsor df to
# standardized data.frame
build_sponsorship <- function(data_json){
    
    sponsor_df <- as.data.frame(data_json$sponsor)
    
    # Members identified by two sets of IDs,
    # bioguide and previously thomas... Need to keep/account for both
    tid    <- sponsor_df$thomas_id
    if(is.null(tid)) tid <- NA_character_
    bid  <- sponsor_df$bioguide_id
    if(is.null(bid)) bid <- NA_character_
    
    sponsor_df <-
        dplyr::mutate(sponsor_df,
                      primary = 1L, withdrawn = 0L,
                      thomas_id = tid, bioguide_id = bid)
    
    # If there were cosponsors, code withdrawal
    cospons <- data_json$cosponsors
    
    if(!is.null(nrow(cospons))){
        tid  <- cospons$thomas_id
        if(is.null(tid)) tid <- rep(NA_character_, nrow(cospons))
        bid  <- cospons$bioguide_id
        if(is.null(bid)) bid <- rep(NA_character_, nrow(cospons))
        
        cospons <-
            data_json$cosponsors %>%
            dplyr::mutate(thomas_id = tid,
                          bioguide_id = bid,
                          primary = 0L,
                          withdrawn = !is.na(withdrawn_at) %>% as.integer(),
                          type = NA_character_)
        
    } else { cospons <- sponsor_df[0, ] }
    
    sponsorship <-
        
        rbind(sponsor_df, cospons[names(sponsor_df)]) %>%
        
        dplyr::mutate(district = lz_pad(district, 2),
                      name  = tolower(name),
                      lname = str_extract(name, "^[^,]+"),
                      fname = str_extract(name, "(?<=,).+"),
                      name  = NULL,
                      bill_id = data_json$bill_id)
    setDT(sponsorship)
    
    setcolorder(sponsorship,
                c("bill_id", "state", "district",
                  "thomas_id", "bioguide_id",
                  "lname", "fname",
                  "title", "type", "primary", "withdrawn"))
    
    return(sponsorship)
}

jslist_to_bill <- function(data_json){
    require(assertthat)
    require(dplyr)
    require(data.table)
    require(coler)
    
    if(has_name(data_json, "amendment_id"))
        data_json <- amend_to_bill(data_json)
    
    # Names of fields to extract
    extr_names <- c("congress", "bill_type", "bill_id",
                    "committees",
                    "official_title", "short_title",
                    "status",
                    "sponsor",
                    "cosponsors",
                    "subjects", "subjects_top_term", "summary", "history",
                    "introduced_at", "updated_at")
    assert_that(is.list(data_json))
    assert_that(has_name(data_json, extr_names))
    
    
    ## sponsorship data ----
    sponsored <- any(!is.na(data_json$sponsor))
    
    if(sponsored){
        sponsorship <- build_sponsorship(data_json)
    } else {
        sponsorship <-
            data.table(bill_id = data_json$bill_id,
                       state = NA_character_, district = NA_character_,
                       thomas_id = NA_character_, bioguide_id = NA_character_,
                       lname = NA_character_, fname = NA_character_,
                       title = NA_character_, type = NA_character_,
                       primary = NA_integer_, withdrawn = NA_integer_)
    }
    
    
    # bill data ----
    history_df <- as.data.frame(data_json$history)
    data <-
        data.table(congress       = data_json$congress,
                   bill_type      = data_json$bill_type,
                   bill_id        = data_json$bill_id,
                   status         = data_json$status,
                   committees     = list(data_json$committees),
                   title_official = data_json$official_title,
                   title          = data_json$short_title,
                   topic_primary  = data_json$subjects_top_term,
                   enacted        = as.integer(history_df$enacted),
                   vetoed         = as.integer(history_df$vetoed),
                   await_sign     = as.integer(history_df$awaiting_signature),
                   intro_date     = as.Date(data_json$introduced_at),
                   upd_date       = as.Date(data_json$updated_at)
        )
    
    # topic data ----
    # some bills have no topic, like hr3-108
    if(!is.na(data_json$subjects_top_term)){
        summarized <- has_name(data_json$summary, "text")
        
        topics <-
            list(bill_id       = data_json$bill_id,
                 primary_topic = data_json$subjects_top_term,
                 committees    = data_json$committees,
                 subjects      = data_json$subjects,
                 summary       = if(summarized) data_json$summary$text else NA
            )
    } else { topics <- list() }
    
    
    structure(lst(bill_id = data_json$bill_id, data, sponsorship, topics),
              class = c("bill"))
    
}

# Helper objects
AgendasProjectTopicCodes <- 
    as.data.table(
        dplyr::tribble( ~ code_major, ~ topic_major,
                    1, "Macroeconomics",
                    2, "Civil Rights",
                    3, "Health",
                    4, "Agriculture",
                    5, "Labor",
                    6, "Education",
                    7, "Environment",
                    8, "Energy",
                    9, "Immigration",
                    10, "Transportation",
                    12, "Law and Crime",
                    13, "Social Welfare",
                    14, "Housing",
                    15, "Domestic Commerce",
                    16, "Defense",
                    17, "Technology",
                    18, "Foreign Trade",
                    19, "International Affairs",
                    20, "Government Operations",
                    21, "Public Lands",
                    23, "Culture"
                    )
    )
