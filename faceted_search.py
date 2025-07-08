import ast


def filter_by_categories(businesses, category_list):
    """
    从business中的"categories"字段中筛选出符合category_list的business_id
    :param businesses: 企业数据
    :param category_list: 分面搜索类别列表
    :return: "categories"字段符合category_list的企业的business_id
    """
    category_list_lower = [cat.lower() for cat in category_list]
    return {
        b["business_id"]
        for _, b in businesses.iterrows()
        if isinstance(b["categories"], str)
        and any(cat.strip().lower() in category_list_lower for cat in ast.literal_eval(b["categories"]))
    }


def filter_businesses(businesses, facets=None):
    """
    分面搜索入口，支持根据city、categories和min_star条件进行分面搜索
    :param businesses: 企业数据
    :param facets: 分面搜索条件，字典结构，{"city": xx, "categories": [yy], "stars": zz}，默认为None
    :return: 符合搜索条件的企业business_id集合，若facets为None则返回所有企业的business_id
    """
    if facets is not None:
        city, categories, min_star = facets.get("city"), facets.get("categories"), facets.get("stars")
    else:
        print("不启用分面搜索！")
        return set(b["business_id"] for _, b in businesses.iterrows())   # 不启用分面搜索，返回所有企业的business_id

    sets = []

    if city is not None:
        sets.append({b["business_id"] for _, b in businesses.iterrows() if b["city"] == city})

    if categories is not None:
        sets.append(filter_by_categories(businesses, categories))

    if min_star is not None:
        sets.append({b["business_id"] for _, b in businesses.iterrows() if b["stars"] >= min_star})

    # 求交集
    if sets:
        return set.intersection(*sets)
    else:
        raise ValueError("传入的facets中参数有误，查询不到对应的企业信息，请检查facets信息！")
