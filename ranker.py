import math
from collections import defaultdict


def score_by_term_frequency(terms, phrases, unigram_index, bigram_index):
    """
    tf检索方法，基于词频进行简单打分
    :param terms: 单词列表
    :param phrases: 短语列表
    :param unigram_index: 单词索引
    :param bigram_index: 双词索引
    :return: 按得分从高到低排列的(review_id, scores)列表
    """
    doc_scores = defaultdict(int)

    # 词项得分
    for term in terms:
        if term in unigram_index:
            for review_id, freq in unigram_index[term].items():
                doc_scores[review_id] += freq

    # 短语得分（权重更高）
    for phrase in phrases:
        if phrase in bigram_index:
            for review_id, freq in bigram_index[phrase].items():
                doc_scores[review_id] += freq * 2

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def score_by_tf_idf(terms, unigram_index, review_df):
    """
    tfidf检索方法，使用TF-IDF进行打分（忽略短语，仅单词）
    :param terms: 单词列表
    :param unigram_index: 单词索引
    :param review_df: 评论数据
    :return: 按得分从高到低排列的(review_id, scores)列表
    """
    doc_scores = defaultdict(float)
    doc_count = len(review_df)

    valid_ids = set(review_df['review_id'])  # 获取有效评论ID

    for term in terms:
        if term in unigram_index:
            df = len(unigram_index[term])  # 包含该词的文档数
            idf = math.log((doc_count + 1) / (df + 1)) + 1  # 避免除0
            for doc_id, tf in unigram_index[term].items():
                if doc_id not in valid_ids:  # 分面搜索中过滤不相关的评论
                    continue
                doc_scores[doc_id] += tf * idf

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def score_by_bm25(query_terms, unigram_index, review_df, k1=1.5, b=0.75):
    """
    bm25检索方法，使用BM25算法进行打分
    :param query_terms: 预处理后的词项列表（只使用 unigram）
    :param unigram_index: 单词索引
    :param review_df: 评论数据
    :param k1: BM25的调节参数，默认为1.5
    :param b: BM25的调节参数，默认为0.75
    :return: 按得分从高到低排列的(review_id, scores)列表
    """
    N = len(review_df)
    doc_lengths = {}
    avgdl = 0   # 语料库中所有评论的平均长度

    valid_ids = set(review_df['review_id'])  # 获取有效评论ID

    # 构造评论长度信息（以 token 数计）
    for _, row in review_df.iterrows():
        doc_id = row['review_id']
        tokens = row['processed_text'].split()
        doc_lengths[doc_id] = len(tokens)
        avgdl += len(tokens)

    avgdl = avgdl / N if N > 0 else 0

    scores = defaultdict(float)

    for term in query_terms:
        if term not in unigram_index:
            continue

        # 该 term 出现在哪些评论中（df）
        posting = unigram_index[term]
        df = len(posting)

        # IDF 计算（加1平滑）
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in posting.items():
            if doc_id not in valid_ids:   # 分面搜索中过滤不相关的评论
                continue
            # 计算bm25得分
            dl = doc_lengths.get(doc_id, 0)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf * tf * (k1 + 1) / denom
            scores[doc_id] += score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

