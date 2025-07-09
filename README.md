# Speech-Integrated Modeling for Behavioral Coding in Counseling

This repository contains the code and resources for the paper:

**"Speech-Integrated Modeling for Behavioral Coding in Counseling"**  
*Do June Min, Verónica Pérez-Rosas, Kenneth Resnicow, Rada Mihalcea*  
*Proceedings of the 26th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL 2025)*

## Abstract

Computational models of psychotherapy often ignore vocal cues by relying solely on text. To address this, we propose MISQ, a framework that integrates speech features directly into language models using a speech encoder and lightweight adapter. MISQ improves behavioral analysis in counseling conversations, achieving ~5% relative gains over text-only or indirect speech methods—underscoring the value of vocal signals like tone and prosody.

## System Overview

Our system combines:
- **Speech Processing**: HuBERT-based audio encoder for extracting speech representations
- **Language Modeling**: MiniChat-2-3B as the base language model with LoRA fine-tuning
- **Multimodal Integration**: Direct integration of audio embeddings with text embeddings
- **Behavioral Analysis**: Focus on motivational interviewing (MI) behavioral codes

## Key Features

- **Multimodality**: Speech and text inputs
- **Two Tasks**: Classification and forecasting.
- **Noise Robustness**: Evaluation under various noise conditions
- **Efficient Training**: LoRA fine-tuning for parameter efficiency

## Repository Structure

```
speech_mi/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config/
│   └── config_full.yaml              # Model configuration
├── data/
│   ├── converted_segmental_information_with_audio_analysis.json
│   └── segmental_information.json    # Dataset information
├── scripts/
│   ├── run_train.sh                   # Training scripts
│   ├── run_test.sh                    # Testing scripts
│   ├── run_noise.sh                   # Noise robustness testing
│   └── continue_train.sh              # Continue training from checkpoint
├── core/
│   ├── run_experiment.py              # Main experiment runner
│   ├── audio_llama.py                 # Audio-integrated LLM
│   ├── audio_encoder.py               # Audio processing module
│   ├── process_data.py                # Data preprocessing
│   └── utils.py                       # Utility functions
├── analysis/
│   ├── plot_confusion.py              # Confusion matrix visualization
│   ├── plot_noise.py                  # Noise robustness analysis
│   └── plot.py                        # General plotting utilities
└── tools/
    ├── qa2_test.py                    # Audio analysis with Qwen2-Audio
    ├── z_diarize.py                   # Speaker diarization
    └── z_transcribe.py                # Audio transcription
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/speech-mi.git
cd speech-mi
```

2. **Create a conda environment**:
```bash
conda create --name speech_mi python=3.11
conda activate speech_mi
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models**:
The system uses the following pre-trained models:
- MiniChat-2-3B (automatically downloaded via transformers)
- HuBERT-large (automatically downloaded via transformers)
- Audio Encoder Weights (Kang et al, 2024)

Download the pretrained audio encoder weights from the following source:

- GitHub Repository: [`wonjune-kang/llm-speech-summarization`](https://github.com/wonjune-kang/llm-speech-summarization)  
- Direct Download: [Google Drive Folder](https://drive.google.com/drive/folders/1o363nAqpyP80tivFNdjmyyoWGCLUeHZS?usp=sharing)

Once downloaded, place the file at the following path in your project directory:


## Quick Start

### 1. Data Preparation

Ensure your data is in the correct format. The system expects:
- `data/converted_segmental_information_with_audio_analysis.json`: Main dataset with audio paths and annotations
- Audio files accessible at the paths specified in the JSON

### 2. Training

Use the provided training scripts:

```bash
# Train all modalities for classification
./run_train.sh

# Or train specific modality
python run_experiment.py --mode train --modality speech \
    --run_name "my_speech_experiment" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 \
    --freeze_encoder --use_lora --use_audio_eos --only_hq_sessions
```

### 3. Testing

Evaluate trained models:

```bash
# Test all trained models
./run_test.sh

# Or test specific model
python run_experiment.py --mode test --modality speech \
    --run_name "test_my_speech_experiment" \
    --test_batch_size 2 --datatype float16 \
    --checkpoint_path "./logs/classification_experiment/my_speech_experiment/checkpoint_epoch_X_step_Y.pt" \
    --lora_checkpoint_path "./logs/classification_experiment/my_speech_experiment/lora_checkpoint_epoch_X_step_Y" \
    --use_lora --use_audio_eos --only_hq_sessions
```

### 4. Noise Robustness Testing

Test model robustness to audio noise:

```bash
./run_noise.sh
```

## Experiment Scripts

### Training Scripts (`run_train.sh`)
- **Classification**: Predicts therapist behaviors and client talk types
- **Forecasting**: Predicts future client responses

### Testing Scripts (`run_test.sh`)
- Evaluates trained models on test sets
- Computes metrics: accuracy, F1-score, BLEU, ROUGE, METEOR, BERTScore
- Generates detailed classification reports

### Noise Robustness (`run_noise.sh`)
- Tests performance under different SNR levels (25, 20, 15, 10, 5, 0 dB)
- Evaluates degradation with noise

### Continue Training (`continue_train.sh`)
- Resume training from checkpoints
- Useful for extending training or hyperparameter tuning

## Configuration

### Key Parameters

- `--modality`: Choose from `speech`, `text`, or `textAA`
- `--task`: Choose from `classification` or `forecasting`
- `--batch_size`: Training batch size (recommended: 2)
- `--learning_rate`: Learning rate (recommended: 1e-4)
- `--freeze_encoder`: Keep audio encoder frozen during training
- `--use_lora`: Enable LoRA fine-tuning
- `--use_audio_eos`: Add end-of-sequence tokens for audio
- `--only_hq_sessions`: Train only on high-quality MI sessions
- `--noise_level`: SNR level for noise robustness testing

### Modalities

1. **Speech**: Direct audio processing with HuBERT encoder
2. **Text**: Text-only baseline (traditional approach)
3. **TextAA (Paralinguistic Captioning)**: Text + audio analysis descriptions

## Results and Analysis

### Visualization Tools

- `plot_confusion.py`: Generate confusion matrices
- `plot_noise.py`: Analyze noise robustness results
- `plot.py`: General visualization utilities

### Output Files

Training and testing generate:
- `test_results.json`: Detailed results and metrics
- `experiment.log`: Training logs
- TensorBoard logs in experiment directories

## Dataset

The system is designed for the AnnoMI dataset but can be adapted for other counseling datasets. Required format:
- Audio files with timestamps
- Transcript with speaker annotations
- Behavioral codes (therapist behaviors, client talk types)
- Session quality ratings

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{min2025paralinguistics,
  author    = {Do June Min and Ver{\'o}nica P{\'e}rez-Rosas and Kenneth Resnicow and Rada Mihalcea},
  title     = {Speech-Integrated Modeling for Behavioral Coding in Counseling},
  booktitle = {Proceedings of the 26th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL 2025)},
  year      = {2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- Do June Min: dojmin@umich.edu

## Acknowledgments

We thank the contributors to the AnnoMI dataset and the open-source community for the tools and models used in this research.