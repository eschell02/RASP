load(file = '/Users/eloiseschell/Desktop/Research/Code and Outputs/SpeakerPickerFolder/IEEE_data_icphs_sharable.rda')

# pick the variables of interest from the table 
subset_df <- subset(pi1, select = c("Speaker_ID", "SpeakerGender", "LD"))

# only include rows where SpeakerGender = M
M_df <- subset_df[subset_df$SpeakerGender == "M", ]

# separate by individual speakers
install.packages("dplyr")
split_data <- split(M_df, M_df$Speaker_ID)

# avg LD of each speaker
average_values <- lapply(split_data, function(subtable) {
  return(mean(subtable$LD))
})

# Convert the list to a data frame
result_df <- data.frame(
  Speaker_ID = names(average_values),
  average_value = unlist(average_values)
)

# List from lowest to highest
ordered_groups <- names(average_values)[order(unlist(average_values))]


