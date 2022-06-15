import logging
import os
import sys

import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


def gbk2utf8():
    file_out = open('data/douluo_utf8.txt', 'w', encoding="utf-8")  # 输出文件路径
    with open('data/douluo.txt', 'r', encoding="GB18030") as file_object:
        for line in file_object:
            line = line.strip()
            file_out.write(line + "\n")
    file_out.close()
    print("end")


def seg_words():
    # 定义一些常量值，多次调用的文件路径放在这里，容易修改
    origin_file = "douluo.txt"  # 初代文件
    stop_words_file = "stop_words.txt"  # 停用词路径
    user_dict_file = "user_dict.txt"  # 用户自定义词典路径
    stop_words = list()
    # 加载停用词
    with open(stop_words_file, 'r', encoding="utf8") as stop_words_file_object:
        contents = stop_words_file_object.readlines()
        for line in contents:
            line = line.strip()
            stop_words.append(line)
    # 加载用户字典
    jieba.load_userdict(user_dict_file)
    target_file = open("douluo_cut_word.txt", 'w', encoding="utf-8")
    with open(origin_file, 'r', encoding="utf-8") as origin_file_object:
        contents = origin_file_object.readlines()
        for line in contents:
            line = line.strip()
            out_str = ''
            word_list = jieba.cut(line, cut_all=False)
            for word in word_list:
                if word not in stop_words:
                    if word != "\t":
                        out_str += word
                        out_str += ' '
            target_file.write(out_str.rstrip() + "\n")
    target_file.close()
    print("end")


def train_model():
    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    # if len(sys.argv) < 4:
    #     print(globals()['__doc__'] % locals())
    #     sys.exit(1)
    # input_dir, outp1, outp2 = sys.argv[1:4]
    # input为输入语料， outp1为输出模型， outp2位vector格式的模型
    input_dir = 'douluo_cut_word.txt'
    outp1 = 'douluo.model'
    outp2 = 'douluo.vector'
    # 训练模型
    # 输入语料目录:PathLineSentences(input_dir)
    # embedding size:256 共现窗口大小:10 去除出现次数5以下的词,多线程运行,迭代10次
    model = Word2Vec(PathLineSentences(input_dir), vector_size=256, workers=5, window=10, min_count=5, epochs=10)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    # 运行命令:输入训练文件目录 python word2vec_model.py data baike.model baike.vector


seg_words()
train_model()
