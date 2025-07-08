import re
from preprocess import preprocess_text
from ranker import score_by_term_frequency, score_by_tf_idf, score_by_bm25
from nltk.corpus import stopwords
from faceted_search import filter_businesses

stop_words = set(stopwords.words('english'))


def parse_query(query_string, process_flag=(True, True, True, True)):
    """
    查询字符串解析函数
    :param query_string: 单个查询字符串
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :return: 预处理后的查询字符串，terms：单词（list类型）；processed_phrases：双引号内的内容（list类型）；sliding_phrases：单词生成的滑动短语（list类型）
    """
    # 提取短语（双引号内内容）
    phrases = re.findall(r'"(.*?)"', query_string)

    # 提取单词
    query_no_phrases = re.sub(r'"(.*?)"', '', query_string)

    processed_query_no_phrases = preprocess_text(query_no_phrases, process_flag=process_flag, stop_words=stop_words)
    processed_phrases = [preprocess_text(p, process_flag=process_flag, stop_words=stop_words)for p in phrases]

    # 分词
    # terms = word_tokenize(processed_query_no_phrases)
    terms = processed_query_no_phrases.split()

    sliding_phrases = [f"{terms[i]} {terms[i + 1]}"for i in range(len(terms) - 1)]   # 对单词生成滑动短语，用于bigram匹配

    return terms, processed_phrases, sliding_phrases


# 执行查询入口函数
def run_query(query_string, unigram_index, bigram_index, method, processed_review_df, business_df, facets=None, top_n=10, process_flag=(True, True, True, True)):
    """
    查询函数入口，进行单条查询检索
    :param query_string: 查询字符串(String类型)
    :param unigram_index: 单词索引
    :param bigram_index: 双词索引
    :param method: 检索方法，可选值为{'tf', 'tfidf', 'bm25'}
    :param processed_review_df: 预处理后的评论数据
    :param business_df: 企业数据
    :param facets: 分面搜索条件，字典结构，{"city": xx, "categories": [yy], "stars": zz}，默认为None
    :param top_n: 返回的评论数量，默认为10
    :param process_flag: 预处理标志，元组结构，为(enable_stemming, ignore_case, process_numbers, remove_punctuation)
                         其中enable_stemming: 是否进行词干提取，默认为True; ignore_case: 是否忽略大小写，默认为True; process_numbers: 是否进行数字处理，True为将整体数字变成单个数字，False为忽略数字，默认为True;
                         remove_punctuation: 是否忽略标点，默认为True
    :return:ranked_docs: 得分最高的top_n个评论的(review_id, score)列表
    """
    # 分面搜索
    filtered_business_ids = filter_businesses(business_df, facets)
    if not filtered_business_ids:
        raise ValueError("分面搜索结果为空，程序终止，请尝试其他分面搜索条件！")
    # 在筛选出的business_id下检索评论
    processed_review_df["review_id"] = processed_review_df["review_id"].astype(str).str.strip()
    filtered_review_df = processed_review_df[processed_review_df["business_id"].isin(filtered_business_ids)]
    filtered_review_ids = set(filtered_review_df["review_id"])

    # 解析查询字符串
    terms, quoted_phrases, sliding_phrases = parse_query(query_string, process_flag=process_flag)
    phrases = quoted_phrases + sliding_phrases
    # 进行查询
    if method == "tf":
        ranked_docs = score_by_term_frequency(terms, phrases, unigram_index, bigram_index)
        ranked_docs = [(rid, score) for rid, score in ranked_docs if rid in filtered_review_ids]   # 只选用符合分面搜索条件的评论
    elif method == "tfidf":
        ranked_docs = score_by_tf_idf(terms, unigram_index, filtered_review_df)
    elif method == "bm25":
        ranked_docs = score_by_bm25(terms, unigram_index, filtered_review_df)
    else:
        raise ValueError(f"未知方法: {method}")
    return ranked_docs[:top_n]


def display_results(ranked_docs, review_df):
    """
    查询结果展示函数
    :param ranked_docs: 查询得到的(review_id, score)列表
    :param review_df: 评论数据
    """
    for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
        text = review_df.loc[review_df['review_id'] == doc_id, 'text'].values[0]
        snippet = text[:200].replace('\n', ' ')   # 选取评论的前200个字符作为摘要
        if doc_id.endswith("?"):
            doc_id = "#Name?"   # 还原原本的“#Name?”
        print(f"[Rank {rank}] ReviewID: {doc_id} | Score: {score}\nSnippet: {snippet}\n")


