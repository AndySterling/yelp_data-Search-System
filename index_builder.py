from collections import defaultdict
import json
import os


def build_unigram_index(df):
    """
    单词索引建立函数
    :param df: 评论数据
    :return: 单词索引，嵌套字典结构，{term: {review_id: 单词在评论中出现的次数}}
    """
    unigram_index = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        review_id = row['review_id']
        tokens = row['processed_text'].split()
        for token in tokens:
            unigram_index[token][review_id] += 1
    result = {term: dict(postings) for term, postings in unigram_index.items()}
    return result


def build_bigram_index(df):
    """
    双词索引建立函数
    :param df: 评论数据
    :return: 单词索引，嵌套字典结构，{biword: {review_id: biword在评论中出现的次数}}
    """
    bigram_index = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        review_id = row['review_id']
        tokens = row['processed_text'].split()
        for i in range(len(tokens) - 1):
            biword = f"{tokens[i]} {tokens[i + 1]}"
            bigram_index[biword][review_id] += 1
    result = {term: dict(postings) for term, postings in bigram_index.items()}
    return result


def build_indexes_and_save(review_df, save_dir='index_output', evaluator_flag=False):
    """
    索引建立入口，建立单/双词索引并保存（非评估模式下）
    :param review_df: 评论数据
    :param save_dir: 索引保存路径，默认为./index_output
    :param evaluator_flag: 是否是评估模式，默认为False
    :return: unigram_index: 单词索引，bigram_index: 双词索引
    """
    unigram_index = build_unigram_index(review_df)
    bigram_index = build_bigram_index(review_df)

    if save_dir is None:
        save_dir = 'index_output'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not evaluator_flag:
        # 非评估模式下保存为JSON文件以便后续使用
        with open(f"{save_dir}/unigram_index.json", 'w', encoding='utf-8') as f:
            json.dump(unigram_index, f, ensure_ascii=False)

        with open(f"{save_dir}/bigram_index.json", 'w', encoding='utf-8') as f:
            json.dump(bigram_index, f, ensure_ascii=False)

        print("索引构建完成并保存。")

    return unigram_index, bigram_index

