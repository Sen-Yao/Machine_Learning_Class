import os.path
import pandas as pd


def read_txt_as_string(file_path):
    try:
        with open(file_path, 'r', encoding='cp1252') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print("File not found.")
        return None


def find_mail_files(path):
    mail_paths = []
    # 遍历当前目录下的所有文件和子目录
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                # 构建完整的文件路径并添加到列表中
                mail_paths.append(os.path.join(root, file))
    return mail_paths


def textParse(bigString):
    """
    接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    """
    import re
    # 使用正则表达式仅匹配英文单词
    listOfTokens = re.findall(r'\b[a-zA-Z]+\b', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(dataSet):
    """
        创建一个包含在所有文档中出现的不重复的词的列表。
    """
    vocabSet = set([])  # create empty set

    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    """
        获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def preprocess():
    ham_mail_paths = find_mail_files('ham')
    spam_mail_paths = find_mail_files('spam')
    ham_strings = []
    spam_strings = []
    for ham_path in ham_mail_paths:
        ham_strings.append(read_txt_as_string(ham_path))
    for spam_path in spam_mail_paths:
        spam_strings.append(read_txt_as_string(spam_path))
    strings = ham_strings + spam_strings
    vocab_list = []
    for long_string in strings:
        vocab_list.append(textParse(long_string))
    vocab_list = createVocabList(vocab_list)

    ham_vectors = []
    spam_vectors = []
    for ham_string in ham_strings:
        ham_vectors.append(bagOfWords2VecMN(vocab_list, textParse(ham_string)))
    for spam_string in spam_strings:
        spam_vectors.append(bagOfWords2VecMN(vocab_list, textParse(spam_string)))
    return ham_vectors, spam_vectors
