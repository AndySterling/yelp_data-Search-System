import os
import time
import nltk
from nltk.corpus import stopwords
from nltk.data import find
import pandas as pd
from preprocess import preprocess_df, calculate_dictionary_size
from index_builder import build_indexes_and_save
from query_processor import run_query, display_results
from evaluator import run_evaluation, save_evaluation_to_csv
import json
import argparse


def str2bool(v):
    """
    字符串转布尔值辅助函数，('yes', 'true', 't', '1')→True；('no', 'false', 'f', '0')→False
    :param v: 待转换的字符串
    :return: 转换后的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (e.g., True/False)")


def download_nltk_resource(resource_id, resource_path=None):
    """
    检查NLTK资源是否存在，若不存在则下载。
    :param resource_id: 资源在nltk.download()中使用的ID
    :param resource_path: 实际资源路径，默认为None，表示自动推断
    """
    try:
        find(resource_path or f"{resource_id}/{resource_id}")
    except LookupError:
        nltk.download(resource_id)


def search_cmd(args):
    """
    检索模式
    :param args: 相关检索参数
    """
    stop_words = set(stopwords.words('english'))

    # 参数解析
    query = args.query
    method = args.method
    top_k = args.top_k
    enable_stemming = args.enable_stemming
    ignore_case = args.ignore_case
    process_numbers = args.process_numbers
    remove_punctuation = args.remove_punctuation
    process_flag = (enable_stemming, ignore_case, process_numbers, remove_punctuation)   # 预处理标志
    review_path = args.review_path
    index_path = args.index_path
    save_dir = args.save_dir
    # 分面搜索字典构建
    facets = {
        "city": args.city,
        "categories": args.categories,
        "stars": args.min_star
    } if any([args.city, args.categories, args.min_star]) else None

    # 加载数据
    business_df = pd.read_json("data/yelp_training_set/yelp_training_set_business.json", lines=True)
    business_df["categories"] = business_df["categories"].apply(lambda x: str(x) if isinstance(x, list) else "[]")
    if review_path:
        processed_review_df = pd.read_csv(review_path, low_memory=False, dtype={"processed_text": "string"})  # 直接加载已有预处理文件
        processed_review_df = processed_review_df.dropna(subset=['processed_text']).copy()
    else:
        review_df = pd.read_json("data/yelp_training_set/yelp_training_set_review.json", lines=True)  # 否则加载原始数据进行数据预处理
        processed_review_df = preprocess_df(review_df, process_flag=process_flag, stop_words=stop_words, evaluator_flag=False)

    # 索引构建
    if index_path:
        # 直接加载已有的单/双词索引
        unigram_index_path = index_path + "/unigram_index.json"
        bigram_index_path = index_path + "/bigram_index.json"
        with open(unigram_index_path, 'r', encoding='utf-8') as f:
            unigram_index = json.load(f)
        with open(bigram_index_path, 'r', encoding='utf-8') as f:
            bigram_index = json.load(f)
    else:
        # 从预处理数据中构建单/双词索引
        processed_review_df['processed_text'] = processed_review_df['processed_text'].fillna('')
        unigram_index, bigram_index = build_indexes_and_save(processed_review_df, save_dir)

    # 查询处理
    print("--------查询处理---------")
    results = run_query(query, unigram_index, bigram_index, method, processed_review_df, business_df, facets=facets,
                        top_n=top_k, process_flag=process_flag)
    # 展示结果
    display_results(results, processed_review_df)


def evaluate_cmd(args):
    """
    评估模式
    :param args: 相关评估参数
    """
    stop_words = set(stopwords.words('english'))

    # 参数解析
    top_k = args.top_k
    # 分面搜索字典构建
    facets = {
        "city": args.city,
        "categories": args.categories,
        "stars": args.min_star
    } if any([args.city, args.categories, args.min_star]) else None
    if facets is not None:
        isfacets = True
    else:
        isfacets = False
    # 获取查询语句列表
    with open(args.query_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    # 加载数据
    review_df = pd.read_json("data/yelp_training_set/yelp_training_set_review.json", lines=True)
    business_df = pd.read_json("data/yelp_training_set/yelp_training_set_business.json", lines=True)
    business_df["categories"] = business_df["categories"].apply(lambda x: str(x) if isinstance(x, list) else "[]")

    # 不同预处理方式下的评估
    print("--------评估模式---------")
    flag_names = ['enable_stemming', 'ignore_case', 'process_numbers', 'remove_punctuation']
    size = [
        {"original_dict_size": 0, "stemmed_dict_size": 0, "case_dict_size": 0, "handle_numbers_dict_size": 0, "remove_punctuation_dict_size": 0}
    ]
    columns = ['stemmed_dict_size', 'case_dict_size', 'handle_numbers_dict_size', 'remove_punctuation_dict_size']
    dict_size_df = pd.DataFrame(size)

    original_dict_size = calculate_dictionary_size(review_df['text'])
    dict_size_df.at[0, "original_dict_size"] = original_dict_size
    print(f"原始字典大小为：{original_dict_size}")
    # 评估流程
    for i, col_name in enumerate(columns):
        process_flag = tuple(j == i for j in range(4))
        # 数据预处理
        print(f"\n当前启用的预处理选项: {flag_names[i]} = True，其余为 False")
        processed_review_df = preprocess_df(review_df, process_flag=process_flag, stop_words=stop_words,
                                                        evaluator_flag=True)
        dict_size = calculate_dictionary_size(processed_review_df['processed_text'])
        dict_size_df.at[0, col_name] = dict_size
        print(f"经过预处理后，字典大小为：{dict_size}")

        # 索引构建
        processed_review_df['processed_text'] = processed_review_df['processed_text'].fillna('')
        unigram_index, bigram_index = build_indexes_and_save(processed_review_df, evaluator_flag=True)
        print(f"单词索引大小为：{len(unigram_index)}, 双词索引大小为：{len(bigram_index)}")

        # 结果评估
        results = run_evaluation(queries, unigram_index, bigram_index, processed_review_df, business_df, top_k=top_k,
                                 facets=facets, process_flag=process_flag)
        save_evaluation_to_csv(results, isfaceted=isfacets, preprocess_flag=process_flag)
        print("-----------------------------------------------------------------")

    # 保存字典大小文件
    local_time = time.localtime()
    formatted = time.strftime("%Y-%m-%d_%H-%M-%S", local_time)
    output_path = f'evaluate/dict_size/output_dict_size_{formatted}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dict_size_df.to_csv(output_path, index=False)
    print(f"字典大小已保存至{output_path}, 评估流程结束")


def main():
    # 资源检查和下载
    download_nltk_resource('punkt', 'tokenizers/punkt')
    download_nltk_resource('stopwords', 'corpora/stopwords')

    parser = argparse.ArgumentParser(description="Yelp评论检索系统CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 子命令：search
    parser_search = subparsers.add_parser('search', help="进行单条查询检索")
    parser_search.add_argument('-q', '--query', type=str, required=True, help="查询字符串")
    parser_search.add_argument('-m', '--method', choices=['tf', 'tfidf', 'bm25'], default='bm25', help="检索方法，可选值为['tf', 'tfidf', 'bm25']， 默认为'bm25'")
    parser_search.add_argument('--city', type=str, default=None, help="分面搜索中的city")
    parser_search.add_argument('--categories', nargs='*', default=None, help="分面搜索中的categories")
    parser_search.add_argument('--min_star', type=float, default=None, help="分面搜索中的min_star")
    parser_search.add_argument('-tk', '--top_k', type=int, default=10, help="返回的评论数量")
    parser_search.add_argument('-es', '--enable_stemming', type=str2bool, default=True, help="预处理标志，是否进行词干提取，默认为True")
    parser_search.add_argument('-ic', '--ignore_case', type=str2bool, default=True, help="预处理标志，是否忽略大小写，默认为True")
    parser_search.add_argument('-pn', '--process_numbers', type=str2bool, default=True, help="预处理标志，是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True")
    parser_search.add_argument('-rp', '--remove_punctuation', type=str2bool, default=True, help="预处理标志，是否忽略标点，默认为True")
    parser_search.add_argument('-r_pth', '--review_path', type=str, help="预处理后评论数据（csv文件）路径，使用此参数可以跳过预处理步骤")
    parser_search.add_argument('-i_pth', '--index_path', type=str, help="索引文件所在目录路径（应包含unigram_index.json和bigram_index.json），使用此参数可以跳过索引构建步骤")
    parser_search.add_argument('-s_dir', '--save_dir', type=str, help="单/双词索引的保存路径，默认为./index_output")
    parser_search.set_defaults(func=search_cmd)

    # 子命令：evaluate
    parser_eval = subparsers.add_parser('evaluate', help="进行批量查询评估")
    # parser_eval.add_argument('--queries', nargs='+', required=True, help="待评估的查询语句列表(list类型)")
    parser_eval.add_argument('-qf', '--query_file', type=str, required=True, help="包含查询语句的txt文件路径，每行对应一条查询语句")
    parser_eval.add_argument('--city', type=str, default=None, help="分面搜索中的city")
    parser_eval.add_argument('--categories', nargs='*', default=None, help="分面搜索中的categories")
    parser_eval.add_argument('--min_star', type=float, default=None, help="分面搜索中的min_star")
    parser_eval.add_argument('-tk', '--top_k', type=int, default=10, help="返回的评论数量")
    parser_eval.set_defaults(func=evaluate_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
