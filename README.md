# Spot-the-Lie-
Spot the Lie is an advanced AI-powered platform designed to combat digital misinformation. By combining the speed of custom-trained Machine Learning models with the deep reasoning capabilities of GPT-4o-mini, this system achieves a remarkable 95% accuracy in identifying fake news and deceptive content.
The project includes several trained model directories:
- `fake_news_model_20250701_180146/`
- `fake_news_model_20250702_124006/`
- `fake_news_model_20250702_124201/`
- `fake_news_model_20250702_125926/`
- `fake_news_model_20250702_130626/`

Each directory contains the trained model files and checkpoints.

## Features

- **DistilBERT-based Classification**: Fast and accurate fake news detection using pre-trained DistilBERT models
- **Hybrid Approach**: Combines local DistilBERT models with OpenAI's GPT models for enhanced accuracy
- **Dataset Integration**: Uses comprehensive fake news dataset for training and evaluation
- **Easy Setup**: Simple virtual environment and dependency management
- **Multiple Model Versions**: Various trained model checkpoints for different use cases

## Notes

- The virtual environment is already set up in the `venv/` directory
- Make sure to activate the virtual environment before running any scripts
- Update the OpenAI API key in `hybrid_modal.py` before using the hybrid approach
- The dataset file should be placed in the parent directory as specified

## Troubleshooting

1. **Virtual Environment Issues**: If activation fails, try recreating the virtual environment
2. **Dependency Issues**: Ensure all packages are installed correctly using `pip install -r requirements.txt`
3. **API Key Issues**: Verify your OpenAI API key is correctly set in `hybrid_modal.py`
4. **Model Loading Issues**: Ensure the model directories are present and accessible

## License

This project is for educational and research purposes. Please ensure compliance with OpenAI's terms of service when using their API. 
