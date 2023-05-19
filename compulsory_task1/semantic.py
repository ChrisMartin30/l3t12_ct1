import spacy
nlp = spacy.load('en_core_web_md')

# Run code extracts from pdf
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(f"{word1} vs {word2}", word1.similarity(word2))
print(f"{word3} vs {word2}", word3.similarity(word2))
print(f"{word3} vs {word1}", word3.similarity(word1))
print()


tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Note what is interesting about similarities between "monkey", "cat", "apple" and "banana".
#
#   "Cat" is more similar to "monkey" than any fruit, which is not surprising as 
#   both are animals, and is roughly equal in similarity to each fruit 
#   
#   "Apple" is more similar to "banana" than either animal, and is roughly equal 
#    in similarity to either animal.
#   
#   "Monkey" is more similar to "cat" then either fruit, but of the fruit it is 
#    markedly more similar to "banana" than "apple", perhaps due to the popular 
#    childhood images of monkeys eating bananas
# 
#   "Banana" is more similar to "apple" than either animal, but as discussed above, 
#    is linked to "monkey" much more than "cat".

# Try other examples
print("\n ------- Own examples ---------")
tokens = nlp('cat car cab dog lion tiger dingo river')

for token1 in tokens:
    for token2 in tokens:
        if token1 != token2:
            print(token1.text, token2.text, token1.similarity(token2))
    print()

# Note about similarities of 'cat car cab dog lion tiger dingo river'
#
# "Cat" is most similar to "dog", another domestic animal, however it is more 
# similar to "dingo" - a wild dog - than either "lion" or "tiger" - both big 
# cats. It is also markedly more similar to "tiger" than "lion". It is not 
# similar to neither "car" nor "cab", despite the very similar spelling. "river" 
# was put in as a relatively unconnected item, which is reflected in the value given
#
# "Car" was most similar to "cab", which presumably reflects the use of cars as 
# taxis or cabs. It was broadly unsimilar to all other items. 
#
# "Cab" was most similar to "car", as mentioned above. It was mostly less similar 
# to other items in the list than the equvalent values for "car"; if a "cab" is a 
# subset of "car" that might account for that. However, "car":"lion" was 
# much lower than "cab":"lion".
#
# "dog" was as mentioned above most similar to "cat". It was more similar to the
#  wild dog "dingo" than either of the big cats, however it was more similar 
# to "tiger" than "lion".
#
# "lion" was most similar to it's fellow big cat the "tiger". It was also more 
# similar to the third wild animal the "dingo" than either of the domestic 
# animals, but of the 2 the "lion" was more similar to the fellow feline 
# "cat" than "dog". It had little similarity to the other objects.
#
# "tiger" had broadly similar results to "lion" which I assume is due to them 
# both being big cats. However, while the order of the two were the same, 
# the "tiger" seemed to have higher similarity scores for each than the "lion"'s score.
#
# "dingo" had the most similarity with "tiger", then "cat" and "lion" before "dog",
#  which is surprising given the "dingo" is a wild dog. It is of low similarity to 
# the other items. 
#
# "river" has low similarity to all items, which is the reason it was put in. It is 
# neither an animal, nor the habitat of a listed animal, nor have similar spelling 
# or purpose to any other list items.


# Run example file with simpler language model and note differences
nlp = spacy.load('en_core_web_sm')
print("\nSimpler model")

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# When comparing the simpler small model to the medium one the 
# first thing to note is that it displays a warning that the small model 
# is not suited for similarity judgements and recommends using a larger model. 
# Otherwise, there are several variations noted, such as 
# the "cat":"apple" being ~ 0.7 in the small model, vs ~ 0.2 in the medium, 
# "apple":"monkey" simple ~0.74 vs medium ~0.23, 
# "apple":"banana" simple ~0.36 vs medium ~0.66  