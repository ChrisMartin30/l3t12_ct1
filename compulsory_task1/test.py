import spacy
# Run example file with simpler language model and note differences
nlp = spacy.load('en_core_web_sm')
print("\nSimpler model")

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


print("\nMedium model")
nlp = spacy.load('en_core_web_md')


tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))