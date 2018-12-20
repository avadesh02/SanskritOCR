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
		self.virama = self.unicodemap[np.where(self.english_input == 'virama')]

	def englishtosanskritunicode(self, word):

		## transliterates english to sanskrit script using an itrans converter
		## returns in Unicode string for Dectionary search api
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
				if self.input_type[np.min(np.where(self.english_input == letter[:-1]))] == 'vowel':
					#print(letter[:-1])
					unicode_index = np.min(np.where(self.english_input == letter[:-1]))
					sanskrit_word = sanskrit_word + self.unicodemap[unicode_index]
				
				else:
					#print(letter[:-1])
					unicode_index = np.min(np.where(self.english_input == letter[:-1]))
				## NOTE: 99 is value for virama
					sanskrit_word = sanskrit_word + self.unicodemap[unicode_index] + self.unicodemap[105]
				
			else:
				if self.input_type[np.min(np.where(self.english_input == letter[:-1]))] == 'vowel':
					sanskrit_word = sanskrit_word[:-6]
					if letter[:-1] == 'a':
						#print(letter[:-1])
						continue
					elif letter[:-1] == 'i':
						#print(letter[:-1])
						# i matra to be added before the word
						unicode_index = np.max(np.where(self.english_input == letter[:-1]))
						for check in range(4,7):
							if sanskrit_word[-check] == '\\':
								sanskrit_word = sanskrit_word[:-check] + self.unicodemap[unicode_index] + sanskrit_word[-check:]
					else:	
						#print(letter[:-1])
						unicode_index = np.max(np.where(self.english_input == letter[:-1]))
						sanskrit_word = sanskrit_word + self.unicodemap[unicode_index]

				else:
					if self.input_type[np.where(self.english_input == letter[:-1])] == 'visarga':
						# Removing halanth for visarga situation
						unicode_index = np.max(np.where(self.english_input == letter[:-1]))
						sanskrit_word = sanskrit_word + self.unicodemap[unicode_index]
					else:
						#print(letter[:-1])
						unicode_index = np.max(np.where(self.english_input == letter[:-1]))
						sanskrit_word = sanskrit_word + self.unicodemap[unicode_index] + self.unicodemap[105]
									
				

		# typecasting to unicode. NOTE: only works in python 3
		sanskrit_word = sanskrit_word[1:]
		return sanskrit_word

	def englishtosanskrit(self,word):
		# function returns unicode form of word for Dictionary search API

		sanskrit_word_uni = self.englishtosanskritunicode(word)
		return sanskrit_word_uni.encode().decode('unicode-escape')

		
