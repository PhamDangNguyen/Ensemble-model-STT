from fastpunct import FastPunct
# The default language is 'english'
fastpunct = FastPunct()
# print(fastpunct.punct([
#                 "john smiths dog is creating a ruccus",
#                 "ys jagan is the chief minister of andhra pradesh",
#                  "we visted new york last year in may"
#                  ]))
                 
# ["John Smith's dog is creating a ruccus.",
# 'Ys Jagan is the chief minister of Andhra Pradesh.',
# 'We visted New York last year in May.']

# punctuation correction with optional spell correction (experimental)

print(fastpunct.punct([
                  'that gives you an idea of what i m up against',
                  'he s looking for you',
                   'he studied at saint john s college cambridge'], correct=True))