import pandas as pd
import numpy as np
import re
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawProtocol
import nltk
from konlpy.tag import Okt

class review(MRJob):
	OUTPUT_PROTOCOL = RawProtocol

	def steps(self):
		return [MRStep(mapper=self.mapper, reducer=self.reducer),MRStep(reducer=self.reduce_sort)]
	
	def mapper(self,_,line):
		_, date, rating, thumbsUp, content = line.split(',')
		okt = Okt()
		nouns = okt.nouns(content)
		for word in nouns:
			if word != '""' and len(word) > 1 and int(rating) <=3:
				yield (word, '1')
	
	def reducer(self, word, counts):
		total_count = sum(int(count) for count in counts)
		yield None, (str(total_count),word)
	
	def reduce_sort(self,_,count_word):
		for count, word in sorted(count_word, key=lambda x: int(x[0]), reverse=True):
			yield word, count
	
if __name__ == '__main__':
	review.run()
