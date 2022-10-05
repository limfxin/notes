# 官网的评论观点抽取
from pprint import pprint
from paddlenlp import Taskflow

schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
ie = Taskflow('information_extraction', schema=schema)
pprint(ie("我去的这个店,服务和卫生都很差,一个菜还100多块钱,性价比可真是高啊"))


#备用的文本
# 我去的这个店，服务和卫生都很差，一个菜还100多块钱，性价比可真是高啊
# 店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队

# 问题：只有局部的判断，没有对于句子整体态度的判断