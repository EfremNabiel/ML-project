## This script builds a spectogram dataset from audiofiles. 
## requires a df with labels and the directory of each audiofile 
## in the working directory.

# loading libraries
library(torch)
library(torchaudio)

## importing data and processing
df <- read.csv("filedir_df.csv")
# removing entries without an associated files
df <- df[!is.na(df$file_dir),]
# restricting analysis to bird calls
df <- df[df$TYPE == "song",]

# converting char time stamp to seconds 
get_seconds <- function(charcode){
  time <- strsplit(charcode, ":")[[1]]
  seconds <- as.numeric(time[1])*60+as.numeric(time[2])
  return(seconds)
}
df$recording_length_seconds <- sapply(df$recording_length, get_seconds)

# only look at clips up to 2 minutes
df <- df[df$recording_length_seconds <= 120,]
# only look at 5 most common genus
top_5_genus <- names(sort(table(factor(df$genus)), decreasing = T)[1:5])
df <- df[df$genus %in% top_5_genus, ]

####  Building spectogram dataset ####
# downsampling factor, quicker processing if audio is downsampled
rs_factor <- 10
# length of each spectogram in pixels
spec_nrows <- 122

# structure to store spectograms and labels in
data <- list()
labels <- c()
# also saving species for potential future use
species <- c()

# iterate through each file and create a spectogram
count = 1
for (i in 1:nrow(df)) {
  skip = FALSE
  # load wave file
  dir <- df$file_dir[i]
  # if torchaudio can't load file, skip it
  tryCatch(
    {
      mp3 <- torchaudio_load(dir)
    }, error = function(e){
      skip = TRUE
    }
  )
  if(skip == T){next}
  # if file is successfully loaded, create spectogram
  else{
    # getting class label
    i_genus <- df$genus[i]
    i_species <- df$scientific_name[i]
    
    # processing into waveform
    waveform_and_sample_rate <- transform_to_tensor(mp3)
    waveform <- waveform_and_sample_rate[[1]]
    sample_rate <- waveform_and_sample_rate[[2]]
    # downsampling
    resampled_rate <- sample_rate / rs_factor
    downsampled_wav <- transform_resample(sample_rate,resampled_rate)(waveform)
    # creating spectogram
    specgram <- transform_spectrogram()(downsampled_wav)
    specgram <- specgram + 0.01
    specgram_as_array <- as.array(specgram$log2()[1]$t())
    
    spec_length <- nrow(specgram_as_array)
    # pad spectogram with 0's if it is too short
    if(spec_length<spec_nrows){
      pad <- matrix(0, nrow = (spec_nrows-spec_length), ncol = 201)
      specgram_as_array <- rbind(specgram_as_array,pad)
    }
    
    # split longer audio files into multiple spectograms of
    # desired length
    J <- floor(spec_length/spec_nrows)
    for (j in 1:J) {
      max((j-1)*spec_nrows,1)
      five_s_gram <- specgram_as_array[(max(((j-1)*spec_nrows)+1,1)):(j*spec_nrows),]
      data[[count]] <- five_s_gram
      labels <- c(labels, i_genus)
      species <- c(species, i_species)
      count <- count + 1
    }
    # print progress 
    print(c(i, count))
  }
}

# store spectograms in array structure which can be used as input
# for model built with keras syntax
array_data <- array(NA, dim = c(length(data), 122, 201))
for(i in 1:length(data)){
  array_data[i,,] <- data[[i]]
}

# encoding labels as numeric
ilabels <- unclass(factor(labels)) - 1
ilabels_species <- unclass(factor(species)) - 1

# creating training and test set
set.seed(1000)
train_index <- sample(seq(1:dim(array_data)[1]), 
                      floor(length(ilabels)*0.80), replace = F)
test_index <- seq(1:dim(array_data)[1])[-train_index]
# randomizing order
set.seed(1000)
test_index <- sample(test_index) 

train_x <- array_data[train_index,,]
test_x <- array_data[test_index,,]

train_y <- ilabels[train_index]
test_y <- ilabels[test_index]

train_genus <- labels[train_index]
test_genus <- labels[test_index]

train_species <- species[train_index]
test_species <- species[test_index]

# saving data
spectogram_data <- list(train_x = train_x,
                        train_y = train_y,
                        test_x = test_x,
                        test_y = test_y,
                        train_genus = train_genus,
                        test_genus = test_genus,
                        train_species = train_species,
                        test_species = test_species)
save(spectogram_data, file = "./spectogram_data.RData")

