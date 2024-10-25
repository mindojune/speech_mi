#!/bin/bash

# Create a directory for re-encoded files
mkdir -p ./data/converted_audios

# Loop through all wav files in the specified directory
for file in ./data/session_audios/*.wav; do
  # Get the base name of the file (without path and extension)
  base_name=$(basename "$file" .wav)
  
  # Use ffmpeg to re-encode the file with 16kHz sample rate and mono channel
  ffmpeg -i "$file" -ar 16000 -ac 1 -c:a pcm_s16le "./data/converted_audios/${base_name}_converted.wav"
done

echo "Re-encoding completed. Check the ./data/converted_audios directory for the new files."
