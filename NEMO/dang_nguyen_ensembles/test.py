from fastpunct import FastPunct
# The default language is 'english'
fastpunct = FastPunct()
print(fastpunct.punct([
                "john smiths dog is creating a ruccus",
                "ys jagan is the chief minister of andhra pradesh",
                 "we visted new york last year in may"
                 ]))
                 
# ["John Smith's dog is creating a ruccus.",
# 'Ys Jagan is the chief minister of Andhra Pradesh.',
# 'We visted New York last year in May.']

# punctuation correction with optional spell correction (experimental)

print(fastpunct.punct([
                  'johns son peter is marring estella in jun',
                  'my name is jun',
                   'kamal hassan is a gud actr'], correct=True))