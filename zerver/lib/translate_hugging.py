import os
from transformers import MarianMTModel, MarianTokenizer
import json


class HuggingTranslator:
    def __init__(self):
        config_filename = 'translation_config.json'
        self.config = self.load_config(config_filename)
        self.tokenizer = MarianTokenizer.frompretrained(self.config['model_name'])
        self.model = MarianMTModel.from_pretrained(self.config['model_name'])

    @staticmethod
    def load_config(self, config_filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_filename)
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def translate(self, text, tgt_lang):
        src_lang = self.detect_language(text)
        inputs = self.tokenizer.prepare_seq2seq_batch([text], src_lang=src_lang, tgt_lang=tgt_lang,
                                                      padding=True, truncation=True)
        translated_ids = self.model.generate(**inputs)
        translated_text = self.tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        return translated_text
