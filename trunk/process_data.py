import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
import re
import os
import time

EMBED_DIM = 200
BiRNN_UNITS = 200

punctuation_dict = {' ': 0, '，': 1, '。': 2, '？': 3}
# punctuation_dict = {' ': 0, '，': 1, '。': 2, '！': 3, '？': 4, '：': 5, "；": 6}
# punctuation_dict = {' ': 0, '|': 1}
punctuation_count = len(punctuation_dict.keys())
# punctuation_count = 2
punctuation_str = ''.join(punctuation_dict.keys()).strip()
punctuation_space_str = ''.join(punctuation_dict.keys())
trainintotal_per=0.8

def clear_data(input_file):
    output_file = input_file+'.clean'
    with open(output_file, 'w', encoding='utf-8') as output:
        with open(input_file, 'r', errors='ignore',encoding='utf-8') as input:
            for line in input:
                #去掉空格
                line = line.strip()
                if line is None:
                    continue
                #得到所有中文标点。去除。？ ，,可能有多个连续的标点
                line=re.sub("[^\u4e00-\u9fa5，。？]+", "" ,line)
                #去除重复标点
                line =re.sub(r"([%s])+" % punctuation_str, r"\1", line)
                if len(line) == 0:
                    continue
                output.write(line + " ")
    return output_file

# 对数据进行格式化,构造训练集和测试集
def format_data(input_file, train_size, words_pre_size):
    train_file = input_file + '.train'
    test_file = input_file + '.test'

    print('train_size: {}'.format(train_size))
    with open(train_file, 'w', encoding='utf-8') as train_f:
        with open(test_file, 'w', encoding='utf-8') as test_f:
            with open(input_file, 'r+', encoding='utf-8') as input_f:
                totaldata=[]
                traindata=[]
                testdata=[]
                input_lines=re.sub('\s', '', input_f.readlines())

                #得到所有中文标点。去除。？ ，,可能有多个连续的标点
                input_lines=re.sub("[^\u4e00-\u9fa5，。？]+", "" ,input_lines)
                #去除重复标点
                input_lines =re.sub(r"([%s])+" % punctuation_str, r"\1", input_lines)

                input_lines_space=''
                for char in input_lines:
                    input_lines_space=char+' '
                #把标点和上一个字符黏在一起，例如 '我 。'变成 '我。'
                f_rule, f_target = (re.compile("\\s*(['，。？])\\s*"), "\g<1> ")
                input_lines = f_rule.sub(f_target, input_lines)
                #按空格分割字符串
                lists=input_lines.split(' ')
                dataall=[]
                for item_s in lists:
                    #判断子字符串里有没有标点
                    if re.compile('.*[{}]'.format(punctuation_str)).match(item_s):
                        #最后一位是标点
                        punctuation = item_s[-1]
                        #是需要统计的标点
                        if punctuation in punctuation_str:
                            label = punctuation_dict[punctuation]
                        else:
                            label = punctuation_dict[' ']

                    else:
                        label = punctuation_dict[' ']
                    templist=[item_s,label]
                    totaldata.append(templist)

                total_count=len(totaldata)
                #按trainintotal_per的比例，分割训练集和测试集数据
                for i in totaldata:
                    if (train_count < (total_count*trainintotal_per-1)):
                        traindata.append(i)
                        train_count=train_count+1
                    else:
                        testdata.append(i)

    word_counts = Counter(row[0].lower() for sample in traindata for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = [0,1,2,3]
    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)
    return traindata, testdata, (vocab, chunk_tags)

#+空格
def AddSpace(x):
    # k = x + ' '
    return x + ' '

def load_data():
    # train = _parse_data(open('data/train_data.data', 'rb'))
    # test = _parse_data(open('data/test_data.data', 'rb'))
    #
    # word_counts = Counter(row[0].lower() for sample in train for row in sample)
    # vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    # chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    #
    # # save initial config data
    # with open('model/config.pkl', 'wb') as outp:
    #     pickle.dump((vocab, chunk_tags), outp)
    #
    # train = _process_data(train, vocab, chunk_tags)
    # test = _process_data(test, vocab, chunk_tags)
    # return train, test, (vocab, chunk_tags)
    start = time.clock()
    print("开始处理数据\n")
    inuptname='100w-zw'
    # modelsavepath='model/'+inuptname +'config.pkl'
    total_datapath = 'data/total_data' + inuptname + '.txt'
    train_datapath = 'data/train_data'+inuptname+'.txt'
    test_datapath = 'data/test_data'+inuptname+'.txt'
    input_file='data/'+inuptname+'.txt'
    #判断训练集和测试集有没有，再决定是否处理原始文本
    if(os.path.exists(train_datapath) and os.path.exists(test_datapath)):
        traindata = pickle.load(open(train_datapath, "rb"))
        testdata = pickle.load(open(test_datapath, "rb"))
    else:
        with open(input_file, 'r', encoding='utf-8') as input_f:
            totaldata = []
            traindata = []
            testdata = []
            train_count=0
            total_count=0
            input_lines=''
            # for line in input_f.readlines():  # 依次读取每行
            #     line = line.strip()
            #     =input_lines+line

            input_lines=''.join(input_f.readlines())
            input_lines = re.sub("[\\s]+", "。", input_lines)
            # 得到所有中文标点。去除。？ ，,可能有多个连续的标点
            input_lines = re.sub("[^\u4e00-\u9fa5，。？]+", "", input_lines)
            # 去除重复标点
            input_lines = re.sub(r"([%s])+" % punctuation_str, r"\1", input_lines)

            #在所有字符后面加空格是为了后面分字以及把标点和前一个字黏在一起
            print("开始每个字后面加空格\n")
            input_lines_list=list(input_lines)
            input_lines=' '.join(input_lines_list)
            print("开始每个字后面加空格完毕\n")

            #。？后面加|，再利用|分句，以免损失标点信息
            print("开始加|\n")
            input_lines = re.sub("[。]+", "。|", input_lines)
            input_lines = re.sub("[？]+", "？|", input_lines)
            print("结束加|\n")

            # 把标点和上一个字符黏在一起，例如 '我 。'变成 '我。'
            f_rule, f_target = (re.compile("\\s*(['，。？！：|])"), "\g<1> ")
            input_lines = f_rule.sub(f_target, input_lines)

            #利用|分句
            list_sen=  input_lines.split('|')
            #删掉最后一个空字符串
            del list_sen[-1]

            #对list_sen内的每个元素按空格分成List，成为单字
            print("开始分字\n")
            lists=[]
            for items in list_sen:
                templist=items.split(' ')
                templist=list(filter(None,templist))
                lists.append(templist)
            print("结束分字\n")

            print("开始生成数据集\n")
            for line_list in lists:
                t_list=[]
                for item_s in line_list:
                    # 判断子字符串里有没有标点
                    if re.compile('.*[{}]'.format(punctuation_str)).match(item_s):
                        # 最后一位是标点
                        punctuation = item_s[-1]
                        # 是需要统计的标点
                        if punctuation in punctuation_str:
                            label = punctuation_dict[punctuation]
                        else:
                            label = punctuation_dict[' ']
                        #去除最后一位的标点
                        item_s=item_s[0:-1]

                    else:
                        label = punctuation_dict[' ']
                    templist = [item_s, label]
                    t_list.append(templist)
                totaldata.append(t_list)
            print("数据集生成完毕\n")

            pickle.dump(totaldata, open(total_datapath, "wb"))
            print("保存数据集\n")

            print("分割训练集和测试集\n")
            total_count = len(totaldata)
            # 按trainintotal_per的比例，分割训练集和测试集数据
            for i in totaldata:
                if (train_count < (total_count * trainintotal_per - 1)):
                    traindata.append(i)
                    train_count = train_count + 1
                else:
                    testdata.append(i)
            print("分割训练集和测试集完毕\n")

    print("计算词汇数量\n")
    word_counts = Counter(row[0].lower() for sample in traindata for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    chunk_tags = [0, 1, 2, 3]
    # save initial config data
    with open('model/config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    #存储traindata，testdata方便下次取用
    if (os.path.exists(train_datapath)==False and os.path.exists(test_datapath)==False):
        pickle.dump(traindata, open(train_datapath, "wb"))
        pickle.dump(testdata, open(test_datapath, "wb"))

    print("将训练集和测试集转换为向量表示\n")
    train = _process_data(traindata, vocab, chunk_tags)
    test = _process_data(testdata, vocab, chunk_tags)

    print("将训练集和测试集转换为向量表示结束\n")
    elapsed = (time.clock() - start)
    print("数据处理一共花费时间（s）：\n", elapsed)
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    s=len(data)
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length
