# 尝试官网程序(信息抽取)
from pprint import pprint
from paddlenlp import Taskflow
schema = ['时间','地点']# Define the schema for entity extraction
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("在10月的一个下午,我在操场上和朋友散步")) # Better print results using pprint

# 备选的主题和文本
#['人名','生病']
# 小红在外出的时候淋了一场雨，最后得了感冒

#['地点'，'事件']
# 在10月的一个下午，我在操场上和朋友散步