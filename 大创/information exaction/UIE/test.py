from pprint import pprint
from paddlenlp import Taskflow
schema = ['人名','关系']
seg = Taskflow("word_segmentation", schema=schema)
pprint(seg("阿诤是全世界最可爱"))