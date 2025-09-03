import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import csv
import html
import re
import numpy as np
from rouge_score import rouge_scorer

class TransformerModel:
    def __init__(self, model_name="distilgpt2"):
        print("загружаем модель distilgpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            print("используется GPU")
        else:
            print("используется CPU")
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            clean_up_tokenization_spaces=True
        )
        
        self.generator.tokenizer.pad_token = self.generator.tokenizer.eos_token
        self.generator.model.config.pad_token_id = self.generator.tokenizer.eos_token_id
        self.generator.model.config.do_sample = True
        self.generator.model.config.temperature = 0.9
        self.generator.model.config.top_k = 50
        self.generator.model.config.top_p = 0.95

    def load_tweets_from_csv(self, file_path):
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            texts = [line.strip() for line in lines if line.strip()]
            print(f"загружено {len(texts)} твитов")
            return texts
        except FileNotFoundError:
            print(f"файл {file_path} не найден!")
            return []
        except Exception as e:
            print(f"ошибка при загрузке файла: {e}")
            return []

    def clean_tweet(self, text):
        if not text or not isinstance(text, str):
            return ""
        
        text = html.unescape(text)
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-z0-9\s\']', ' ', text)

        replacements = {
            r"\bdon t\b": "don't", r"\bdidn t\b": "didn't", r"\bwon t\b": "won't",
            r"\bcant\b": "can't", r"\bwouldnt\b": "wouldn't", r"\bshouldnt\b": "shouldn't",
            r"\bwasnt\b": "wasn't", r"\bhas nt\b": "hasn't", r"\bhavent\b": "haven't",
            r"\bdo not\b": "don't", r"\bcan not\b": "can't", r"\bwould not\b": "wouldn't",
            r"\bdont\b": "don't", r"\bwont\b": "won't", r"\bim\b": "i'm", r"\bit s\b": "it's"
        }
        
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)

        text = re.sub(r"(?<!n)'(?!t\b)", " ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def prepare_data(self, file_path='raw_data.csv'):
        print("загружаем и подготавливаем данные")
        texts = self.load_tweets_from_csv(file_path)

        if not texts:
            print("создаем демонстрационные данные")
            texts = [
                "I love to play football with my friends",
                "Today is a beautiful day for a walk",
                "I can't believe how fast time flies",
                "You should try this amazing recipe",
                "We are going to the beach tomorrow"
            ]

        cleaned_texts = [self.clean_tweet(t) for t in texts if t and t.strip()]
        filtered_texts = [t for t in cleaned_texts if 3 <= len(t.split()) <= 20]

        print(f"очищено текстов: {len(cleaned_texts)}")
        print(f"отфильтровано текстов: {len(filtered_texts)}")

        val_size = min(150, int(0.1 * len(filtered_texts)))
        test_size = min(150, int(0.1 * len(filtered_texts)))
        val_texts = filtered_texts[:val_size]
        test_texts = filtered_texts[val_size:val_size + test_size]

        print(f"валидационная выборка: {len(val_texts)} текстов")
        
        return val_texts, test_texts

    def validate_model(self, val_texts):
        print("-" * 40)
        print("валидация модели")
        print("-" * 40)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        examples = []

        seq_len = 5

        print("выполняем оценку модели на валидационной выборке")

        for i, text in enumerate(val_texts[:100]):       
            words = text.split()
            if len(words) < 3:
                continue
            
            context_words = words[:min(seq_len, len(words)-1)]
            context_text = " ".join(context_words).strip()
            true_word = words[min(seq_len, len(words)-1)]
            
            try:
                result = self.generator(
                    context_text,
                    max_new_tokens=10,
                    num_return_sequences=1,
                    truncation=True,
                    pad_token_id=self.generator.tokenizer.eos_token_id
                )
                
                generated_text = result[0]['generated_text'].strip()
                new_text = generated_text[len(context_text):].strip()
                pred_word = new_text.split()[0] if new_text.split() else ""
                
            except Exception as e:
                pred_word = ""
            
            if pred_word:
                score = scorer.score(true_word, pred_word)
                rouge_l = score['rougeL'].fmeasure
                rouge_scores.append(rouge_l)
                
                if len(examples) < 5 and rouge_l > 0:
                    examples.append({
                        "context": context_text,
                        "true": true_word,
                        "pred": pred_word,
                        "rouge": rouge_l
                    })

        avg_rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0
        std_rouge_l = np.std(rouge_scores) if rouge_scores else 0.0

        print("\n" + "-" * 40)
        print("результаты валидации")
        print("-" * 40)
        print(f"метрика: ROUGE-L (точность предсказания следующего слова)")
        print(f"количество оценок: {len(rouge_scores)}")
        print(f"средний ROUGE-L: {avg_rouge_l:.4f} ± {std_rouge_l:.4f}")
        if rouge_scores:
            print(f"медиана ROUGE-L: {np.median(rouge_scores):.4f}")

        if examples:
            print("\nлучшие примеры предсказаний:")
            print("-" * 40)
            for i, ex in enumerate(examples, 1):
                print(f"пример {i}:")
                print(f"контекст:    '{ex['context']}'")
                print(f"истинное:    '{ex['true']}'")
                print(f"предсказанное: '{ex['pred']}'")
                print(f"ROUGE-L:     {ex['rouge']:.4f}")
                print()
        else:
            print("\nне удалось получить качественные примеры предсказаний")
            
        return avg_rouge_l, examples

    def generate_texts(self, train_texts):
        print("-" * 40)
        print("генерация текстов")
        print("-" * 40)

        good_prompts = [
            t for t in train_texts 
            if len(t.split()) >= 3 and any(t.startswith(x) for x in ['i ', 'you ', 'we ', 'they ', 'today', 'im ', 'youre ', 'its '])
        ]

        good_prompts = []
        for t in train_texts:
            if len(t.split()) >= 3 and any(t.startswith(x) for x in ['i ', 'you ', 'we ', 'they ', 'today', 'im ', 'youre ', 'its ']):
                words = t.split()
                # берем первые 3/4 слов из предложения
                cut_index = max(2, int(len(words) * 0.75))  # минимум 2 слова оставляем
                cut_prompt = ' '.join(words[:cut_index])
                good_prompts.append(cut_prompt)

        if not good_prompts:
            good_prompts = [
                "i love to",
                "today is a",
                "you should try",
                "we are going",
                "im feeling very"
            ]

        demo_prompts = good_prompts[:3] if len(good_prompts) >= 3 else good_prompts

        print("генерируем тексты на основе промптов")
        print("-" * 40)

        results = []
        for i, prompt in enumerate(demo_prompts, 1):
            try:
                # result = self.generator(
                #     prompt,
                #     max_new_tokens=30,
                #     do_sample=True,
                #     temperature=0.8,
                #     top_p=0.9,
                #     num_return_sequences=1,
                #     pad_token_id=self.generator.tokenizer.eos_token_id,
                #     truncation=True
                # )
                result = self.generator(
                    prompt,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                    truncation=True,
                    no_repeat_ngram_size=2
                )
                
                generated_text = result[0]['generated_text'].strip()
                generated_part = generated_text.split('\n')[0].strip()
                clean_output = generated_text[len(prompt):].strip()

                # if '.' in clean_output:
                #     clean_output = clean_output.split('.')[0] + '.'
                # elif '!' in clean_output:
                #     clean_output = clean_output.split('!')[0] + '!'
                # else:
                #     clean_output = clean_output.split('\n')[0].strip()
                
                # if clean_output:
                #     # ищем первую точку, восклицательный или вопросительный знак
                #     for end_char in ['.', '!', '?']:
                #         if end_char in clean_output:
                #             end_index = clean_output.index(end_char) + 1
                #             clean_output = clean_output[:end_index].strip()
                #             break

                sentences = re.split(r'(?<=[.!?])\s+', clean_output)
                if sentences:
                    # берем первое предложение, которое не пустое и не просто точка
                    first_sentence = sentences[0].strip()
                    if first_sentence and first_sentence != '.':
                        clean_output = first_sentence
                    elif len(sentences) > 1:
                        clean_output = sentences[1].strip()
                    else:
                        clean_output = clean_output.split('\n')[0].strip()
                else:
                    clean_output = clean_output.split('\n')[0].strip()

                print(f"промпт {i}:")
                print(f"  вход:  {prompt}")
                print(f"  выход: {clean_output}\n")
                print(f"  полностью: {generated_part}\n")
                print("-" * 40)
                
                
            except Exception as e:
                print(f"ошибка при генерации для промпта '{prompt}': {e}")
                print("-" * 40)
                results.append({"prompt": prompt, "error": str(e)})

        print("\nгенерация завершена")
        return results
    
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU")