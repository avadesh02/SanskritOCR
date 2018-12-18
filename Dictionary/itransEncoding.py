completeDict = {"0915":"k","0916":"kh","0917":"g","0918":"gh","0919":"N^","091a":"ch","091b":"chh","091c":"j","091d":"jh","091e":"JN","091f":"T","0920":"Th","0921":"D","0922":"Dh","0923":"N","0924":"t","0925":"th","0926":"d","0927":"dh","0928":"n","0929":"$","092a":"p","092b":"ph","092c":"b","092d":"bh","092e":"m","092f":"y","0930":"r","0931":"$","0932":"l","0933":"ll","0934":"lll","0935":"v","0936":"sh","0937":"shh","0938":"s","0939":"h","093e":"aa","093f":"i","0940":"ii","0941":"u","0942":"uu","0943":"R^i","0944":"R^I","0945":"*","0946":"*","0947":"e","0948":"ai","0949":"*","094a":"*","094b":"o","094c":"au","0905":"a","0906":"aa","0907":"i","0908":"ii","0909":"u","090a":"uu","090b":"R^i","090c":"L^i","090f":"e","0910":"ai","0913":"o","0914":"au","0901":".N","0902":".n","0903":"H","0904":"*","090d":"*","090e":"*","0911":"*","0912":"*","093c":".","093d":".a","094d":".h","0950":"AUM","0966":"0","0967":"1","0968":"2","0969":"3","096a":"4","096b":"5","096c":"6","096d":"7","096e":"8","096f":"9"}

# word is the input
'''
word = "\u092a+\u0905+\u0920+\u0905"
letters = word.split("+")

modWord = []
for i in range(0,len(letters)):
    modWord.append('0' + hex(ord(letters[i])).split('x')[-1])

modWord = [completeDict[x] for x in modWord]
modWord = ''.join(str(e) for e in modWord)
'''

# If input is given from code wriiten in textconverter then
'''

letters = word.split("\\u")[1:]
modWord = [completeDict[x.lower()] for x in letters]
modWord = ''.join(str(e) for e in modWord)

'''

print(modWord) 			# modWord is word transformed to Encoding needed for dictionary

#code for reading Sanskrit-English dictionary
with open("dictall.txt") as f:	
	content = f.readlines()

content = [x.strip() for x in content]

content = [x.split("=") for x in content]
	
words = [x[0].strip() for x in content]
meaning = [x[1].strip() for x in content]

sanskritDict = dict.fromkeys(words)

for i in range(0,len(content)):
	sanskritDict[words[i]] = meaning[i]

# meaning of Word
print(sanskritDict[modWord])		
