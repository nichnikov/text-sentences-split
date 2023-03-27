import re
import torch

model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                  model='silero_te')
# input_text = input('She heard Missis Gibson talking on in a sweet monotone, and wished to attend to what she was saying, but the Squires visible annoyance struck sharper on her mind.\n')
#text = "Об установлении социальных Норм отпуска топлива реализуемого населению топливоснабжающими организациями\n"
# text = "ПОСТАНОВЛЕНИЕ от 31 декабря 2015 г. N 1379 ОБ УТВЕРЖДЕНИИ СХЕМЫ РАЗМЕЩЕНИЯ НЕСТАЦИОНАРНЫХ ТОРГОВЫХ ОБЪЕКТОВ НА ТЕРРИТОРИИ ГОРОДА РОСТОВА-НА-ДОНУ\n"
text = "ПОСТАНОВЛЕНИЕ от 31 декабря 2015 г. N 1379 ОБ УТВЕРЖДЕНИИ СХЕМЫ РАЗМЕЩЕНИЯ НЕСТАЦИОНАРНЫХ ТОРГОВЫХ ОБЪЕКТОВ НА ТЕРРИТОРИИ ГОРОДА РОСТОВА-НА-ДОНУ"

input_text = re.sub('[^\w\b\s,.]', " ", text)
output_text = apply_te(input_text.lower(), lan='ru')
print(f"Input: \n{input_text}\nOutput:\n{output_text}")