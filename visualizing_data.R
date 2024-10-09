# using torchaudio to process mp3s
library(torchaudio)

# function to transform images into audio wave array and spectrogram
transform_data <- function(dir){
  # load audio file from directory
  mp3 <- torchaudio_load(dir)
  # transform to audio wave
  tens <- transform_to_tensor(mp3)
  wav <- tens[[1]]
  sr <- tens[[2]]
  downsampled_wav <- transform_resample(sr,(sr/10))(wav)
  wav_as_array <- as.array(downsampled_wav[1,])
  # transform to spectrogram on log scale
  spec <- transform_spectrogram()(downsampled_wav) + 0.01
  logspec <- spec$log2()[1]$t()
  logspec_as_array <- as.array(logspec)
  # output transformed data
  output <- list(wav = wav_as_array,
                 spec = logspec_as_array)
  return(output)
}

# loading to mp3s as examples of data used in project
crypt348data <- transform_data("./data/mp3s/Crypturellus348.mp3")
orta1362data <- transform_data("./data/mp3s/Ortalis1362.mp3")

# example figures used in presentation
par(mfrow = c(2,1))
plot(ts(crypt348data$wav), xlab = "t", ylab = "frequency", main = "Crypturellus")
plot(ts(orta1362data$wav), xlab = "t", ylab = "frequency", main = "Ortalis")

par(mfrow = c(2,1))
image(crypt348data$spec, main = "Crypturellus")
image(orta1362data$spec, main = "Ortalis")

# example figures used in report
par(mfrow = c(1,1))
image(crypt348data$spec[1:122,], main = "Crypturellus")

plot(ts(crypt348data$wav[1:22000]), xlab = "t", ylab = "frequency", main = "Crypturellus")
