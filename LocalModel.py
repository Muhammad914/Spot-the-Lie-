import openai 
import logging
import os
import torch
import numpy as np
import re
import requests
from typing import Dict, Any, Optional
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model import NewsDetector, clean_text
from tenacity import retry, stop_after_attempt, wait_fixed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from transformers.trainer import Trainer
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
import datetime
import os
os.environ["OMP_NUM_THREADS"] = "4"  # CPU threads limith
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPEN_AI_API_KEY = ""



class NewsValidator:
    def __init__(self):
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _check_url(self, url: str) -> str:
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.text

    def validate_news(self, text: str) -> float:
        urls = re.findall(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]"
            r"|(?:%[0-9a-fA-F][0-9a-f-F]))+",
            text,
        )
        credibility_score = 0.0

        if urls:
            for url in urls[:2]:
                try:
                    # html = self._check_url(url)
                    domain = url.split("/")[2]
                    logger.info(f" Checking domain: {domain}")

                    credible_domains = [
                        "reuters.com",
                        "bbc.com",
                        "aljazeera.com",
                        "nytimes.com",
                        "dawn.com",
                    ]

                    fake_domains = [
                        "infowars.com",
                        "naturalnews.com",
                        "yournewstube.com",
                        "worldnewsdailyreport.com",
                        "beforeitsnews.com",
                    ]

                    if any(cd in domain for cd in credible_domains):
                        credibility_score += 0.3
                        logger.info(f" Found credible domain: {domain}")

                    if any(fd in domain for fd in fake_domains):
                        credibility_score -= 0.5
                        logger.info(f" Found fake domain: {domain}")

                except Exception as e:
                    logger.warning(f" Error processing URL {url}: {e}")

        logger.info(f"URL Credibility Score: {credibility_score}")
        return credibility_score


def find_data_file() -> Optional[str]:
    """Find the fake news dataset file"""
    possible_locations = ["./data", "./", "../data", "../"]
    filename = "train.csv"

    for location in possible_locations:
        file_path = os.path.join(location, filename)
        if os.path.exists(file_path):
            logger.info(f"Found data file at: {file_path}")
            return file_path

    logger.error(f"Could not find data file '{filename}'")
    logger.info(
        "Please place your fake_news_dataset.csv file in the current directory or ./data/ folder"
    )
    return None
def load_data(custom_label_map=None):
    """Load and preprocess the fake news dataset."""
    file_path = find_data_file()
    if not file_path:
        logger.error("Data file not found.")
        return None

    df = None
    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        logger.error("Failed to read CSV with available encodings.")
        return None

    # Handle combined column
    if 'text;label' in df.columns:
        df[['text', 'label']] = df['text;label'].str.split(';', n=1, expand=True)
        df.drop(columns=['text;label'], inplace=True)

    # Ensure required columns exist
    if 'label' not in df.columns or 'text' not in df.columns:
        logger.error(f"Required columns missing. Found: {df.columns.tolist()}")
        return None

    # Default label mappings
    default_mappings = [
        {'FALSE': 0, 'TRUE': 1},
        {'fake': 0, 'real': 1},
        {'false': 0, 'true': 1},
        {'0': 0, '1': 1},
        {0: 0, 1: 1}
    ]
    label_map = custom_label_map or default_mappings
    if isinstance(label_map, dict):
        label_map = [label_map]

    # Apply label mapping
    for mapping in label_map:
        try:
            df['label'] = df['label'].map(mapping).astype(int)
            if set(df['label'].dropna().unique()) == {0, 1}:
                logger.info(f"Applied mapping: {mapping}")
                break
        except Exception:
            continue
    else:
        logger.error(f"Label mapping failed. Unique values: {df['label'].unique()}")
        return None

    # Drop rows with missing data
    df.dropna(subset=['label', 'text'], inplace=True)
    if df.empty:
        logger.error("Dataframe empty after cleaning.")
        return None

    # Clean text
    df['text'] = df['text'].apply(lambda x: clean_text(x, capitalize=False))

    # Filter out short text
    df = df[df['text'].str.len() > 50]

    # Shuffle and reset index
    return df[['text', 'label']].sample(frac=1, random_state=42).reset_index(drop=True)





class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def train_model(hyperparams=None) -> Optional[str]:
    logger.info(" Starting model training...")
    df = load_data()
    if df is None:
        logger.error(" Failed to load data for training")
        return None

    logger.info(f" Dataset loaded: {len(df)} samples")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.1,
        random_state=42,
        stratify=df["label"],
    )

    logger.info(" Initializing tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=300
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=300
    )

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # type: ignore

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_save_dir = f"./fake_news_model_{timestamp}"
    os.makedirs(local_save_dir, exist_ok=True)

    default_hyperparams = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size":  4,
        "learning_rate": 3e-5,
    }
    hyperparams = {**default_hyperparams, **(hyperparams or {})}

    logger.info(" Starting training...")
    training_args = TrainingArguments(
        output_dir=os.path.join(local_save_dir, "results"),
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,
        logging_steps=20,
        report_to=None, 
        **hyperparams,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    final_model_path = os.path.join(local_save_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f" Model saved to {final_model_path}")
    return final_model_path


def check_model_completeness(model_path: str) -> bool:
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]
    missing_files = []

    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        logger.warning(f"Model incomplete. Missing files: {missing_files}")
        return False

    logger.info(" Model directory is complete")
    return True


def find_or_train_model() -> Optional[str]:
    """Find existing model or train a new one"""
    model_dirs = [d for d in os.listdir(".") if d.startswith("fake_news_model_")]

    if model_dirs:
        model_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)

        for model_dir in model_dirs:
            model_path = os.path.join(".", model_dir, "final_model")
            if os.path.exists(model_path) and check_model_completeness(model_path):
                logger.info(f" Found complete model: {model_path}")
                return model_path
            else:
                logger.warning(f" Incomplete model found: {model_path}")

    logger.info("No complete model found. Training new model...")
    return train_model()


class HybridNewsDetector:
    def __init__(self, distilbert_model_path: str = "", openai_api_key: str = ""):
        """
        Initialize Hybrid News Detector combining DistilBERT and OpenAI
        """
        self.validator = NewsValidator()

        self.distilbert_detector: Optional[NewsDetector] = None
        if distilbert_model_path and os.path.exists(distilbert_model_path):
            try:
                self.distilbert_detector = NewsDetector(distilbert_model_path)
                if self.distilbert_detector.model:
                    logger.info("DistilBERT model loaded successfully")
                else:
                    logger.warning(
                        "DistilBERT model failed to load properly, will use OpenAI only"
                    )
                    self.distilbert_detector = None
            except Exception as e:
                logger.error(f"Failed to load DistilBERT model: {e}")
                logger.warning("Will use OpenAI only")
                self.distilbert_detector = None
        else:
            logger.warning("DistilBERT model path not found, will use OpenAI only")

        self.openai_api_key = openai_api_key or OPEN_AI_API_KEY
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.openai_model = "gpt-4o-mini"
        logger.info("Hybrid News Detector initialized successfully")

    def _get_openai_prediction(self, news_text: str) -> Dict[str, Any]:
        """Get prediction from OpenAI GPT-3.5-turbo"""
        try:
            prompt = f'"{news_text}" — just tell me in a single word: either Real or Fake.'

            logger.info(f" OpenAI Analysis - Input: '{news_text}'")

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a news fact-checking expert. Analyze the given "
                            "news text and respond with only 'Real' or 'Fake' based "
                            "on your assessment of its credibility and factual accuracy."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            raw_content = response.choices[0].message.content or ""
            prediction = raw_content.strip().lower()
            logger.info(f"OpenAI Raw Response: '{raw_content.strip()}'")

            if prediction in ["real", "true", "legitimate", "credible"]:
                logger.info("OpenAI Prediction: REAL (confidence: 85%)")
                return {"prediction": "Real", "confidence": 0.85}
            elif prediction in ["fake", "false", "untrue", "misleading"]:
                logger.info("OpenAI Prediction: FAKE (confidence: 85%)")
                return {"prediction": "Fake", "confidence": 0.85}
            else:
                logger.warning(
                    f" OpenAI Unclear Response: '{prediction}' - "
                    "Defaulting to Real (confidence: 50%)"
                )
                return {"prediction": "Real", "confidence": 0.5}

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"prediction": "Error", "confidence": 0.0, "error": str(e)}

    def _get_distilbert_prediction(
        self, news_text: str, credibility_score: float = 0.0
    ) -> Dict[str, Any]:
        if not self.distilbert_detector or not self.distilbert_detector.model:
            logger.warning("DistilBERT model not loaded")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "error": "DistilBERT model not loaded",
            }

        try:
            logger.info(f" DistilBERT Analysis - Input: '{news_text}'")
            result = self.distilbert_detector.predict_news(news_text)

            if "error" in result:
                logger.error(f"DistilBERT Error: {result['error']}")
                return result

            pred = 0 if result["prediction"] == "Real" else 1
            confidence = float(result["confidence"])

            if (pred == 0 and credibility_score > 0) or (
                pred == 1 and credibility_score < 0
            ):
                adjusted_confidence = min(
                    confidence * (1 + abs(credibility_score) * 0.2), 1.0
                )
                logger.info(
                    f"Credibility boost applied: {confidence:.1%} → "
                    f"{adjusted_confidence:.1%}"
                )
            else:
                adjusted_confidence = max(
                    confidence * (1 - abs(credibility_score) * 0.1), 0.1
                )
                logger.info(
                    f" Credibility penalty applied: {confidence:.1%} → "
                    f"{adjusted_confidence:.1%}"
                )

            adjusted_confidence = min(max(adjusted_confidence, 0.1), 1.0)

            logger.info(
                f" DistilBERT Prediction: {result['prediction']} "
                f"(confidence: {adjusted_confidence:.1%})"
            )
            logger.info(
                f" DistilBERT Credibility Score: {credibility_score:.2f}"
            )

            return {
                "prediction": result["prediction"],
                "confidence": adjusted_confidence,
                "raw_confidence": confidence,
                "credibility_score": credibility_score,
                "text_snippet": result.get(
                    "text_snippet",
                    news_text[:100] + "..." if len(news_text) > 100 else news_text,
                ),
            }
        except Exception as e:
            logger.error(f"DistilBERT prediction error: {e}")
            return {"prediction": "Error", "confidence": 0.0, "error": str(e)}

    def predict_news(self, news_text: str) -> Dict[str, Any]:
        """
        Get hybrid prediction: First check with DistilBERT, if Fake then verify with OpenAI
        """
        logger.info(f"Starting Hybrid Analysis: {news_text[:100]}...")
        logger.info("=" * 60)

        logger.info(" STEP 0: URL Validation & Credibility Scoring")
        credibility_score = self.validator.validate_news(news_text)
        logger.info(f"Overall Credibility Score: {credibility_score}")
        logger.info("=" * 60)

        logger.info("STEP 1: DistilBERT Analysis")
        distilbert_result = self._get_distilbert_prediction(
            news_text, credibility_score
        )

        if "error" in distilbert_result:
            logger.warning("DistilBERT failed, falling back to OpenAI only")
            logger.info(" STEP 2: OpenAI Fallback Analysis")
            openai_result = self._get_openai_prediction(news_text)

            if "error" in openai_result:
                logger.error("Both models failed!")
                return {
                    "prediction": "Error",
                    "error": (
                        "Both models failed: DistilBERT - "
                        f"{distilbert_result['error']}, OpenAI - "
                        f"{openai_result['error']}"
                    ),
                    "confidence": 0.0,
                    "model_used": "both_failed",
                    "text_snippet": news_text[:100] + "..."
                    if len(news_text) > 100
                    else news_text,
                }

            logger.info("Using OpenAI result as fallback")
            return {
                **openai_result,
                "model_used": "openai_only",
                "distilbert_error": distilbert_result["error"],
                "credibility_score": credibility_score,
                "text_snippet": news_text[:100] + "..."
                if len(news_text) > 100
                else news_text,
            }

        distilbert_pred = distilbert_result["prediction"]
        distilbert_conf = float(distilbert_result["confidence"])

        logger.info(
            f"DistilBERT Final Result: {distilbert_pred} "
            f"({distilbert_conf:.1%})"
        )

        # Verification conditions
        should_verify = True

        if not should_verify:
            logger.info(
                "DistilBERT predicts REAL with high confidence - "
                "No verification needed"
            )
            logger.info("=" * 60)
            return {
                "prediction": "Real",
                "confidence": distilbert_conf,
                "model_used": "distilbert_only",
                "distilbert_prediction": distilbert_pred,
                "distilbert_confidence": distilbert_conf,
                "credibility_score": credibility_score,
                "verification_needed": False,
                "text_snippet": news_text[:100] + "..."
                if len(news_text) > 100
                else news_text,
            }

        if distilbert_pred == "Fake":
            verification_reason = "DistilBERT predicts FAKE"
        elif distilbert_conf < 0.60:
            verification_reason = (
                f"DistilBERT confidence too low ({distilbert_conf:.1%})"
            )
        else:
            verification_reason = (
                "DistilBERT Real prediction needs verification "
                f"(confidence: {distilbert_conf:.1%})"
            )

        logger.info(f"🔍 {verification_reason} - Starting OpenAI verification")
        logger.info("STEP 2: OpenAI Verification")
        openai_result = self._get_openai_prediction(news_text)

        if "error" in openai_result:
            logger.warning(" OpenAI verification failed, using DistilBERT result")
            logger.info("=" * 60)
            return {
                "prediction": distilbert_pred,
                "confidence": distilbert_conf,
                "model_used": "distilbert_only",
                "distilbert_prediction": distilbert_pred,
                "distilbert_confidence": distilbert_conf,
                "credibility_score": credibility_score,
                "openai_error": openai_result["error"],
                "verification_needed": True,
                "verification_failed": True,
                "text_snippet": news_text[:100] + "..."
                if len(news_text) > 100
                else news_text,
            }

        openai_pred = openai_result["prediction"]
        openai_conf = float(openai_result["confidence"])

        logger.info(
            f"OpenAI Final Result: {openai_pred} ({openai_conf:.1%})"
        )
        logger.info("STEP 3: Model Comparison")

        if distilbert_pred == openai_pred:
            final_confidence = (distilbert_conf + openai_conf) / 2
            logger.info(f"MODELS AGREE: Both predict {distilbert_pred}")
            logger.info(
                f"Final Confidence: {final_confidence:.1%} (average)"
            )
            logger.info("=" * 60)
            return {
                "prediction": distilbert_pred,
                "confidence": final_confidence,
                "model_used": "hybrid_agreement",
                "distilbert_prediction": distilbert_pred,
                "distilbert_confidence": distilbert_conf,
                "openai_prediction": openai_pred,
                "openai_confidence": openai_conf,
                "credibility_score": credibility_score,
                "verification_needed": True,
                "models_agree": True,
                "verification_reason": verification_reason,
                "text_snippet": news_text[:100] + "..."
                if len(news_text) > 100
                else news_text,
            }

        # MODELS DISAGREE
        if distilbert_pred == "Fake" and openai_pred == "Real":
            final_confidence = openai_conf * 0.8 + distilbert_conf * 0.2
            final_prediction = "Real"
            logger.info(
                " DistilBERT=Fake, OpenAI=Real → Trusting OpenAI (80% weight)"
            )
        elif distilbert_pred == "Real" and openai_pred == "Fake":
            final_confidence = openai_conf * 0.8 + distilbert_conf * 0.2
            final_prediction = "Fake"
            logger.info(
                " DistilBERT=Real, OpenAI=Fake → Trusting OpenAI (80% weight)"
            )
        else:
            final_confidence = openai_conf * 0.7 + distilbert_conf * 0.3
            final_prediction = (
                "Real" if openai_conf > distilbert_conf else "Fake"
            )
            logger.info("Weighted Decision: OpenAI 70%, DistilBERT 30%")

        logger.warning(
            f"MODELS DISAGREE: DistilBERT={distilbert_pred}, "
            f"OpenAI={openai_pred}"
        )
        logger.info(
            f"Final Prediction: {final_prediction} "
            f"(confidence: {final_confidence:.1%})"
        )
        logger.info("=" * 60)

        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "model_used": "hybrid_disagreement",
            "distilbert_prediction": distilbert_pred,
            "distilbert_confidence": distilbert_conf,
            "openai_prediction": openai_pred,
            "openai_confidence": openai_conf,
            "credibility_score": credibility_score,
            "verification_needed": True,
            "models_agree": False,
            "verification_reason": verification_reason,
            "text_snippet": news_text[:100] + "..."
            if len(news_text) > 100
            else news_text,
        }


def interactive_hybrid_detection(
    distilbert_model_path: Optional[str] = None,
) -> None:
    """
    Interactive mode for hybrid news detection - matches modal.py style
    """
    detector = HybridNewsDetector(distilbert_model_path or "")

    if not detector.distilbert_detector or not detector.distilbert_detector.model:
        print("\n--- OpenAI-Only Fake News Detection System ---")
        print("DistilBERT model unavailable, using OpenAI GPT-3.5-turbo")
        print("Enter news text or 'quit' to exit\n")
    else:
        print("\n--- Hybrid Fake News Detection System ---")
        print("Enhanced Verification Logic:")
        print("   • DistilBERT analysis first")
        print("   • OpenAI verification when:")
        print("     - DistilBERT predicts Fake")
        print("     - DistilBERT confidence < 60%")
        print("     - DistilBERT predicts Real but confidence < 70%")
        print("Enter news text or 'quit' to exit\n")

    while True:
        text = input("Enter news text: ") or ""

        if text.lower().strip() == "quit":
            break

        if not text.strip():
            print("Please enter some text to analyze.\n")
            continue

        result = detector.predict_news(text)

        print("\n" + "=" * 60)
        print(" DETECTION RESULTS")
        print("=" * 60)

        if "error" in result:
            print(f" Error: {result['error']}")
        else:
            print(f"FINAL PREDICTION: {result['prediction']}")
            print(f"CONFIDENCE: {result['confidence']:.1%}")
            print(f" MODEL USED: {result['model_used']}")
            if "credibility_score" in result:
                print(
                    f"CREDIBILITY SCORE: {result['credibility_score']:.2f}"
                )
            print()

            if "distilbert_prediction" in result:
                print("DISTILBERT ANALYSIS:")
                print(f" Prediction: {result['distilbert_prediction']}")
                print(
                    f" Confidence: "
                    f"{result['distilbert_confidence']:.1%}"
                )
                if "credibility_score" in result:
                    print(
                        f"   Credibility Score: "
                        f"{result['credibility_score']:.2f}"
                    )
                if "distilbert_error" in result:
                    print(f"   Error: {result['distilbert_error']}")
                print()

            if result.get("verification_needed", False):
                print("VERIFICATION DETAILS:")
                if "verification_reason" in result:
                    print(f"    Reason: {result['verification_reason']}")
                if result.get("verification_failed", False):
                    print("  OpenAI Verification: Failed")
                    print("   Using DistilBERT result as fallback")
                elif "openai_prediction" in result:
                    print(
                        f"   OpenAI Prediction: {result['openai_prediction']}"
                    )
                    print(
                        f"   OpenAI Confidence: "
                        f"{result['openai_confidence']:.1%}"
                    )
                    if result.get("models_agree", False):
                        print("   Both models agree")
                    else:
                        print(
                            "    Models disagree - weighted decision used"
                        )
                        if (
                            "openai_confidence" in result
                            and "distilbert_confidence" in result
                        ):
                            if (
                                result["openai_confidence"]
                                > result["distilbert_confidence"]
                            ):
                                print(
                                    "    Weighting: OpenAI 80%, "
                                    "DistilBERT 20% (OpenAI higher confidence)"
                                )
                            else:
                                print(
                                    "    Weighting: OpenAI 70%, "
                                    "DistilBERT 30% (standard)"
                                )
                print()

            print(f"TEXT ANALYZED: {result['text_snippet']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Starting the Hybrid Fake News Detection Application...")
    print("=" * 60)

    model_path = find_or_train_model()

    if model_path:
        print(f"Using model: {model_path}")
        interactive_hybrid_detection(model_path)
    else:
        print(
            " Failed to find or train a model. "
            "Please check your dataset and try again."
        )
