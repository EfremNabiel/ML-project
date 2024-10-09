library(torch)
library(torchaudio)

df <- read.csv("filedir_df.csv")


# removing entries without an associated files
df <- df[!is.na(df$file_dir),]

# only consider bird songs
df <- df[df$TYPE == "song",]

# converting char time stamp to seconds 
get_seconds <- function(charcode){
  time <- strsplit(charcode, ":")[[1]]
  seconds <- as.numeric(time[1])*60+as.numeric(time[2])
  return(seconds)
}
df$recording_length_seconds <- sapply(df$recording_length, get_seconds)

# only look at clips up to five minutes so that building dataset is faster
df <- df[df$recording_length_seconds <= (2*60),]
# only look at 5 most common genus
top_5_genus <- names(sort(table(df$genus), decreasing = T)[1:5])
df <- df[df$genus %in% top_5_genus, ]


####  Building rnn dataset ####
# downsampling factor, quicker processing if audio is downsampled
rs_factor <- 10
# 5 second wave
wav_ncols <- (44000/10)*(5.45)

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
    wav_as_array <- as.array(downsampled_wav)
    
    if(dim(wav_as_array)[1] == 2){
      wav_as_array <- wav_as_array[1,]
    }
    wav_length <- length(wav_as_array)
    
    if(wav_length<wav_ncols){
      pad <- rep(0, (wav_ncols-wav_length))
      wav_as_array <- c(wav_as_array,pad)
    }
    
    # split longer audio files into multiple audiowaves of
    # desired length
    J <- floor(wav_length/wav_ncols)
    for (j in 1:J) {
      max((j-1)*wav_ncols,1)
      short_wav <- wav_as_array[(max(((j-1)*wav_ncols)+1,1)):(j*wav_ncols)]
      data[[count]] <- short_wav
      labels <- c(labels, i_genus)
      species <- c(species, i_species)
      count <- count + 1
    }
    
    # print progress 
    print(c(i, count))
  }
}

array_data <- array(NA, dim = c(length(data), 1, wav_ncols))
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
rnn_data <- list(train_x = train_x,
                        train_y = train_y,
                        test_x = test_x,
                        test_y = test_y,
                        train_genus = train_genus,
                        test_genus = test_genus,
                        train_species = train_species,
                        test_species = test_species)
save(rnn_data, file = "./rnn_data.RData")



