## Contains code for converting english to devanagari and vice versa
## Author : Avadesh
## Date : Dec 17th 2018

import numpy as np
import os 
import sys
import pandas

class textconverter:
	
	def __init__(self):
		self.itransdict = pandas.read_excel('itransdict.xlsx') 
		self.english_input = self.itransdict['INPUT'].values
		self.input_type = self.itransdict['INPUT-TYPE'].values
		self.unicodemap = self.itransdict['#sanskrit'].values

	def englishtosanskrit(self, word):

		## transliterates english to sanskrit script using an itrans converter
		sanskrit_word = 'u'
		## The hash(#) symbol is added to avoid bugs with the last letter combinations. and # is not in the list.
		## TODO: find a better algorithm to deal with the last letter issues
		word = word + '#'
		while len(word) > 1:
			word_length = len(word)
			letter_index = 0
			flag = 1
			letter = word[letter_index]	
			# identifying the letter combination to be converted
			while flag:
				if letter in self.english_input:
					if word_length > 1:
						letter_index = letter_index + 1
						if letter_index < word_length:	
							letter = letter + word[letter_index]
				else:		
					flag = 0
					word = word[letter_index:]
	
		# index value of english input to convert to unicode form
			if len(sanskrit_word) < 2:
				unicode_index = np.min(np.where(self.english_input == letter[:-1]))
				sanskrit_word = sanskrit_word + self.unicodemap[unicode_index]
			else:
				if letter[:-1] == 'a':
					# Nothing is required to be added
					continue
				elif letter[:-1] == 'i':
					# i matra to be added before the word
					unicode_index = np.max(np.where(self.english_input == letter[:-1]))
					for check in range(4,7):
						if sanskrit_word[-check] == '\\':
							sanskrit_word = sanskrit_word[:-check] + self.unicodemap[unicode_index] + sanskrit_word[-check:]
				else:	
					unicode_index = np.max(np.where(self.english_input == letter[:-1]))
					sanskrit_word = sanskrit_word + self.unicodemap[unicode_index]
				

		# typecasting to unicode. NOTE: only works in python 3
		sanskrit_word = sanskrit_word[1:]
		return sanskrit_word



### Test ###

#word = input("enter an english form of a sanskrit word to convert\n")
word = 'iimli'
tt = textconverter()
word = tt.englishtosanskrit(word)
print(word)
print(word.encode().decode('unicode-escape'))

		
