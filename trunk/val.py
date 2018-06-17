import bilsm_crf_model
import process_data
import numpy as np
import re

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
predict_text = '针对一些在生活中孩子遇到的不常见字，为了方便阅读，我们都加以拼音标注，这样就克服了小朋友自助阅读的障碍，有利于他们快速正确的阅读'
#去掉输入文本的所有标点
predict_text = re.sub("[^\u4e00-\u9fa5]+", "", predict_text)
print(predict_text)
str, length = process_data.process_data(predict_text, vocab)
model.load_weights('model/crf.h5')
raw = model.predict(str)[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

per, loc, org = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t==1:
        per=per+s+'，'
    elif t==2:
        per = per + s + '。'
    elif t==3:
        per = per + s + '？'
    else:
        per=per+s
    #
    #     per += ' ' + s if (t == 0) else s
    # if t in (2, 1):
    #     org += ' ' + s if (t == 2) else s
    # if t in (3, 2):
    #     loc += ' ' + s if (t == 3) else s

print("预测结果:\n" + per)
