#!/usr/bin/env python
#coding=utf-8

import os

# 数据集路径
data_file = './data/spam.csv'

# 预处理后的数据集
proc_data_file = os.path.join('./data/proc_spam.csv')

# 文本类别字典
text_type_dict = {'ham': 1,
                  'spam': 0}
