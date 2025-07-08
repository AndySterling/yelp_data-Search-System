from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import string


def calculate_dictionary_size(text_column):
    """
    字典大小计算函数
    :param text_column: 待计算的文本内容
    :return: 若flag为True，则返回字典大小，否则返回0
    """
    # 合并所有文本，进行分词并计算唯一单词
    all_text = " ".join(text_column)
    words = word_tokenize(all_text)
    unique_words = set(words)
    return len(unique_words)


def text_stemming(text, enable_stemming=True):
    """
    词干提取函数
    :param text: 待处理的文本
    :param enable_stemming: 是否进行词干提取，默认为True
    :return: 若enable_stemming为True，则返回词干提取后的文本，否则返回分词后的文本
    """
    if isinstance(text, str):
        words = word_tokenize(text)  # 如果传入的是字符串，表明未进行分词处理，分词
    else:
        words = text

    if enable_stemming:
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]  # 词干提取
    return " ".join(words)


def case_handling(text, ignore_case=True):
    """
    大小写处理函数
    :param text: 待处理的文本
    :param ignore_case: 是否进行大小写处理，默认为True
    :return: 若ignore_case为True，则返回大小写处理后的文本，否则返回原始文本
    """
    if ignore_case:
        return text.lower()
    return text


def handle_numbers(text, process_numbers=False):
    """
    数字处理函数
    :param text: 待处理的文本
    :param process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True
    :return: 数字处理后的文本
    """
    if process_numbers:
        # 将整体数字转为单个数字的形式
        text = re.sub(r'\d+', lambda x: ' '.join(list(x.group())), text)
    else:
        # 忽略数字
        text = re.sub(r'\d+', '', text)
    return text


def handle_punctuation(text, remove_punctuation=True):
    """
    标点处理函数
    :param text: 待处理的文本
    :param remove_punctuation: 是否进行标点处理，默认为True
    :return: 若remove_punctuation为True，则返回标点处理后的文本，否则返回原始文本
    """
    if remove_punctuation:
        return text.translate(str.maketrans('', '', string.punctuation))
    return text


def preprocess_text(data, process_flag=(True, True, True, True), stop_words=None):
    """
    查询字符串数据预处理函数
    :param data: 查询字符串
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :param stop_words: 停用词，若不为None则进行停用词过滤，默认为None
    :return: 处理后的文本
    """
    text = data
    enable_stemming, ignore_case, process_numbers, remove_punctuation = process_flag

    text = case_handling(text, ignore_case)
    text = handle_numbers(text, process_numbers)
    text = handle_punctuation(text, remove_punctuation)
    words = text.split()
    if stop_words is not None:
        words = [w for w in words if w not in stop_words]
    text = text_stemming(words, enable_stemming)
    return text


def preprocess_df(data, process_flag=(True, True, True, True), stop_words=None, evaluator_flag=False):
    """
    原始数据预处理函数
    :param data: 原始数据，即评论内容
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :param stop_words: 停用词，若不为None则进行停用词过滤，默认为None
    :param evaluator_flag: 是否为评估模式，默认为False
    :return: review_df：预处理后的评论数据
    """
    if isinstance(data, pd.DataFrame):
        review_df = data.copy()
        enable_stemming, ignore_case, process_numbers, remove_punctuation = process_flag

        # 简单的数据预处理，对review_id列删除缺失值和非#NAME?的重复值，对#NAME?，将其变成Unknown_i?（其中#Name?每出现一次，i加1）
        review_df = review_df[review_df['review_id'].notna()]
        mask = review_df['review_id'] == '#NAME?'   # 找出review_id是#NAME?的行
        name_indices = review_df[mask].index   # 获取#Name?对应的索引
        for i, idx in enumerate(name_indices, start=1):
            review_df.at[idx, 'review_id'] = f"Unknown_{i}?"
        review_df = review_df.drop_duplicates(subset='review_id')

        if enable_stemming or ignore_case or process_numbers or remove_punctuation:
            print("*************")
            print(f"数据预处理方式为：\n词干提取：{enable_stemming}\n忽略大小写：{ignore_case}\n数字处理(True为将整体数字变成单个数字，False为忽略数字)：{process_numbers}\n忽略标点：{remove_punctuation}")
            print("*************")
            review_df['processed_text'] = review_df['text']

            review_df['processed_text'] = review_df['processed_text'].apply(
                lambda row: case_handling(row, ignore_case))

            review_df['processed_text'] = review_df['processed_text'].apply(
                lambda row: handle_numbers(row, process_numbers))

            review_df['processed_text'] = review_df['processed_text'].apply(
                lambda row: handle_punctuation(row, remove_punctuation))

            # 停用词过滤
            review_df['processed_text'] = review_df['processed_text'].apply(
                lambda row: preprocess_text(row, process_flag=(False, False, False, False), stop_words=stop_words))

            review_df['processed_text'] = review_df['processed_text'].apply(
                lambda row: text_stemming(row, enable_stemming))

        else:
            review_df.rename(columns={'text': 'processed_text'}, inplace=True)
            print("未启用任何数据预处理")
        if not evaluator_flag:
            # 非评估模式下保存预处理结果
            review_df.to_csv('output_review.csv', index=False, encoding='utf-8')
            print("数据预处理结果已保存至 output_review.csv")

        return review_df
