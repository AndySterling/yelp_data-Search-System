import pandas as pd
import time
from faceted_search import filter_businesses
from query_processor import parse_query, run_query
import os
import math


def precision(retrieved, relevant):
    """
    精确率计算函数
    :param retrieved: 检索到的评论列表
    :param relevant: 真实相关的评论列表
    :return: 精确率
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    tp = len(retrieved_set & relevant_set)   # 检索到且真实相关的评论
    return tp / len(retrieved) if retrieved else 0


def recall(retrieved, relevant):
    """
    召回率计算函数
    :param retrieved: 检索到的评论列表
    :param relevant: 真实相关的评论列表
    :return: 召回率
    """
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    tp = len(retrieved_set & relevant_set)   # 检索到且真实相关的评论
    return tp / len(relevant) if relevant else 0


def f1_score(prec, rec):
    """
    F1分数计算函数
    :param prec: 精确率
    :param rec: 召回率
    :return: F1分数
    """
    if prec + rec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)


def generate_relevance_judgments(queries, review_df, business_df, facets=None, top_k=None, process_flag=(True, True, True, True)):
    """
    伪相关文档生成函数。对每个查询使用相同的预处理方法，对评论文本进行处理后，判断是否包含全部查询词项，若包含，则该评论为伪相关文档
    :param queries: 查询字符串列表
    :param review_df: 评论数据
    :param business_df: 企业数据
    :param facets: 分面搜索条件，字典结构，{"city": xx, "categories": [yy], "stars": zz}，默认为None
    :param top_k: 每个查询最多选多少条相关评论，默认值为None，函数将返回全部相关评论
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :return: 查询的相关文档，字典结构，{query_string: [relevant_review_ids]}
    """
    # 分面搜索
    filtered_business_ids = filter_businesses(business_df, facets)
    if not filtered_business_ids:
        raise ValueError("分面搜索结果为空，程序终止，请尝试其他分面搜索条件！")
    # 在筛选出的business_id检索找评论
    filtered_review_df = review_df[review_df["business_id"].isin(filtered_business_ids)]
    relevance_judgments = {}

    for query in queries:
        # 解析查询字符串
        processed_query_terms, quoted_phrases, sliding_phrases = parse_query(query, process_flag=process_flag)
        matched_docs = []
        for _, row in filtered_review_df.iterrows():
            # 单词匹配，若查询字符串中75%的单词在评论中出现，则认为单词匹配成功
            text_lower = row['processed_text'].lower()
            term_match_count = sum(word.lower() in text_lower for word in processed_query_terms)
            term_match = term_match_count >= max(1, math.floor(len(processed_query_terms) * 0.75))
            # 短语匹配，如果评论中出现一个短语，则认为短语匹配成功
            phrase_match = any(phrase.lower() in text_lower for phrase in quoted_phrases)

            if term_match or phrase_match:
                matched_docs.append(row['review_id'])   # 当上述两种匹配模式满足其一时，认为该评论是相关文档

            if top_k is not None and len(matched_docs) >= top_k:
                break
        relevance_judgments[query] = matched_docs

    return relevance_judgments


def evaluate_query(query, relevant_docs, unigram_index, bigram_index, review_df, business_df, top_k=10, facets=None, method='bm25', process_flag=(True, True, True, True)):
    """
    单个查询字符串评估函数
    :param query: 查询字符串(String类型)
    :param relevant_docs: 该查询字符串对应的相关评论，列表结构
    :param unigram_index: 单词索引
    :param bigram_index: 双词索引
    :param review_df: 评论数据
    :param business_df: 企业数据
    :param top_k: 每个查询最多选多少条相关评论
    :param facets: 分面搜索条件，字典结构，{"city": xx, "categories": [yy], "stars": zz}，默认为None
    :param method: 检索方法，默认为''bm25'，可选值为{'tf', 'tfidf', 'bm25'}
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :return: 精确率prec，召回率rec，F1分数f1
    """
    # 获取检索到的评论
    ranked_docs = run_query(query, unigram_index, bigram_index, method, review_df, business_df, facets=facets, top_n=top_k, process_flag=process_flag)
    retrieved = [review_id for review_id, _ in ranked_docs]
    # 评估指标计算
    prec = precision(retrieved, relevant_docs)
    rec = recall(retrieved, relevant_docs)
    f1 = f1_score(prec, rec)

    return prec, rec, f1


def run_evaluation(sample_queries, unigram_index, bigram_index, review_df, business_df, top_k=10, facets=None, process_flag=(True, True, True, True)):
    """
    评估函数入口
    :param sample_queries: 待评估的查询语句列表(list类型)
    :param unigram_index: 单词索引
    :param bigram_index: 双词索引
    :param review_df: 评论数据
    :param business_df: 企业数据
    :param top_k: 每个查询最多选多少条相关评论
    :param facets: 分面搜索条件，字典结构，{"city": xx, "categories": [yy], "stars": zz}，默认为None
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :return: 评估结果，字典结构，{method: [{'query': query, 'precision': prec, 'recall': rec, 'f1': f1}]}
    """
    # 生成伪相关文档
    relevance_judgments = generate_relevance_judgments(sample_queries, review_df, business_df=business_df, facets=facets, top_k=None, process_flag=process_flag)
    methods = ['tf', 'tfidf', 'bm25']
    results = {m: [] for m in methods}

    for method in methods:
        print(f"\nEvaluating method: {method.upper()}")
        for query in sample_queries:
            relevant = relevance_judgments[query]
            prec, rec, f1 = evaluate_query(query, relevant, unigram_index, bigram_index, review_df, business_df, top_k=top_k, facets=facets, method=method, process_flag=process_flag)
            results[method].append({'query': query, 'precision': prec, 'recall': rec, 'f1': f1})
            print(f"run_evaluation: Query: {query}\nPrecision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}\n")

    return results


def save_evaluation_to_csv(results, save_dir='evaluate', isfaceted=False, preprocess_flag=(True, True, True, True)):
    """
    评估结果保存函数，将评估结果输出到csv文件
    :param results: 评估结果
    :param save_dir: 保存文件路径，默认为./evaluate，若开启分面搜索，则默认为./evaluate/faceted
    :param isfaceted: 是否开启分面搜索，默认为False
    :param preprocess_flag: 元组 (enable_stemming, ignore_case, process_numbers, remove_punctuation)
    """
    dir = save_dir if not isfaceted else save_dir + '/faceted'

    if not os.path.exists(dir):
        os.mkdir(dir)

    local_time = time.localtime()
    formatted = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)

    preprocess_str = (
        f"enable_stemming={preprocess_flag[0]}, ignore_case={preprocess_flag[1]}, process_numbers={preprocess_flag[2]}, remove_punctuation={preprocess_flag[3]}"
        if any(preprocess_flag) else "None"
    )

    for method, metrics in results.items():
        df = pd.DataFrame(metrics)
        df['preprocess'] = preprocess_str   # 保存预处理方式
        file_path = os.path.join(dir, f'eval_{method}_{formatted}.csv')
        df.to_csv(file_path, index=False)
        print(f"结果已保存至 {file_path}")



