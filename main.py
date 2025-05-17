import pandas as pd
import pyarrow.parquet as pq
import json
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import ast
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix
import seaborn as sns

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置显示选项（按需调整参数）
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', None)  # 不限制列宽

# 读取 Parquet 文件
parquet_file_ = r'30G_data_new\part-00000.parquet'
df = pd.read_parquet(parquet_file_).head(1000)
df = df.drop(
    columns=["id", "last_login", "user_name", "fullname", "email", "age", "income", "gender", "country", "address",
             "is_active", "phone_number"])

# 读取 mode.json 文件
with open(r'product_catalog.json\product_catalog.json', 'r',
          encoding='utf-8') as f:
    mode_data = json.load(f)

# 创建商品 ID 到类别的映射
id_to_category = {product['id']: product['category'] for product in mode_data['products']}


# 提取 purchase_history 字段中的商品ID
def extract_product_ids(row):
    try:
        purchase_history = ast.literal_eval(row['purchase_history'])
        if 'items' in purchase_history:
            return [item['id'] for item in purchase_history['items'] if 'id' in item]
        else:
            return []
    except Exception as e:
        print(f"解析错误: {e}")
        return []


df['product_ids'] = df.apply(extract_product_ids, axis=1)


# 提取 purchase_history 字段中的商品价格
def extract_avg_price(row):
    try:
        purchase_history = ast.literal_eval(row['purchase_history'])
        if "avg_price" in purchase_history:
            return [purchase_history["avg_price"]]
        else:
            return []
    except Exception as e:
        print(f"解析错误: {e}")
        return []


df['avg_price'] = df.apply(extract_avg_price, axis=1)


# 将商品 ID 转换为商品类别
def convert_to_categories(product_ids):
    return [id_to_category.get(id) for id in product_ids if id in id_to_category]


df['categories'] = df['product_ids'].apply(convert_to_categories)
# 在转换商品ID为类别后添加以下代码
# print("转换后的 categories 示例:\n", df['categories'].head())


### 任务1：完善商品类别关联规则挖掘（调整参数并优化筛选逻辑）
if df['categories'].apply(len).sum() > 0:
    # --- 新增代码：过滤低频类别 ---
    all_categories = [category for sublist in df['categories'] for category in sublist]
    category_counts = Counter(all_categories)
    filtered_categories = [cat for cat, count in category_counts.items() if count >= 1]
    df['categories'] = df['categories'].apply(lambda x: [cat for cat in x if cat in filtered_categories])

    # 生成布尔型 DataFrame
    categories_dummies = pd.get_dummies(df['categories'].explode()).groupby(level=0).max().astype(bool)
    # print("独热编码后的列数:", len(categories_dummies.columns))

    if len(categories_dummies.columns) > 0:
        # 调整到任务要求的参数
        frequent_itemsets = apriori(categories_dummies, min_support=0.000002, use_colnames=True)

        if not frequent_itemsets.empty:
            # 修正项集格式
            frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
                lambda x: frozenset([str(item) for item in x]))

            # 调整到任务要求的置信度阈值
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000005)

            # 增强电子产品规则筛选
            electronics = ['智能手机', '笔记本电脑', '平板电脑', '智能手表', '耳机', '音响', '相机', '摄像机', '游戏机']
            electronics_rules = rules[
                (rules['antecedents'].apply(lambda x: any(item in electronics for item in x))) |
                (rules['consequents'].apply(lambda x: any(item in electronics for item in x)))
                ]
            # 添加规则有效性检查
            if not electronics_rules.empty:
                print("\n电子产品关联规则TOP5:")
                print(electronics_rules[['antecedents', 'consequents', 'support', 'confidence']].head())
            else:
                print("\n未发现显著的电子产品关联规则")

            # 可视化商品类别关联规则
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='support', y='confidence', size='lift', data=rules)
            plt.title('商品类别关联规则')
            plt.xlabel('支持度')
            plt.ylabel('置信度')
            plt.savefig('dog1.png')
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='support', y='confidence', size='lift', data=electronics_rules)
            plt.title('电子产品与其他类别之间的规则')
            plt.xlabel('支持度')
            plt.ylabel('置信度')
            plt.savefig('dog_ele.png')
            plt.show()
        else:
            print("\n未找到满足最小支持度的频繁项集，请降低支持度阈值。")
    else:
        print("\n无法生成类别哑变量，请检查数据。")
else:
    print("\n所有行的类别数据均为空，请检查数据。")

### 任务2：完善支付方式关联分析（新增高价值商品分析）
if 'categories_filtered' not in df.columns:
    # 过滤低频类别（确保与主流程一致）
    all_categories = [category for sublist in df['categories'] for category in sublist]
    category_counts = Counter(all_categories)
    filtered_categories = [cat for cat, count in category_counts.items() if count >= 1]
    df['categories_filtered'] = df['categories'].apply(lambda x: [cat for cat in x if cat in filtered_categories])


def analyze_payment_methods(df):
    try:
        # 确保输入数据包含必要字段
        if 'purchase_history' not in df.columns or 'categories_filtered' not in df.columns:
            print("缺少必要字段: purchase_history 或 categories_filtered")
            return pd.DataFrame()

        # 解析支付方式
        df['purchase_history_parsed'] = df['purchase_history'].apply(ast.literal_eval)
        df['payment_method'] = df['purchase_history_parsed'].apply(lambda x: x.get('payment_method', 'unknown'))

        # 仅保留高频支付方式（降低阈值）
        payment_counts = df['payment_method'].value_counts()
        frequent_payments = payment_counts[payment_counts >= 1].index.tolist()  # 出现至少1次即可
        df_filtered = df[df['payment_method'].isin(frequent_payments)].copy()

        # 生成支付方式与类别的组合
        df_filtered['payment_category'] = df_filtered.apply(
            lambda row: [f"{row['payment_method']}_{cat}" for cat in row['categories_filtered']],
            axis=1
        )

        # 展开组合并生成布尔矩阵
        payment_dummies = pd.get_dummies(df_filtered['payment_category'].explode()).groupby(level=0).max().astype(bool)

        # 使用FP-growth算法（降低支持度阈值）
        payment_frequent_itemsets = fpgrowth(payment_dummies, min_support=0.000001, use_colnames=True)

        # 新增高价值商品分析
        if 'avg_price' in df.columns:
            # 展开价格列表并转换为数值
            df['avg_price'] = df['avg_price'].apply(lambda x: x[0] if x else 0)
            high_value_df = df[df['avg_price'] > 5000]

            if not high_value_df.empty:
                # 分析高价值商品支付方式
                preferred_payment = high_value_df['payment_method'].value_counts().idxmax()
                print(f"\n高价值商品(>5000)首选支付方式: {preferred_payment}")

                # 高价值商品支付方式关联分析
                high_value_dummies = pd.get_dummies(high_value_df['payment_method']).astype(bool)
                hv_frequent = fpgrowth(high_value_dummies, min_support=0.000001, use_colnames=True)

                if not hv_frequent.empty:
                    hv_rules = association_rules(hv_frequent, metric="confidence", min_threshold=0.6)
                    print("\n高价值商品支付方式规则:")
                    print(hv_rules[['antecedents', 'consequents', 'support', 'confidence']])

        # 调整支付方式关联规则参数
        payment_frequent_itemsets = fpgrowth(payment_dummies, min_support=0.000001, use_colnames=True)
        payment_rules = association_rules(payment_frequent_itemsets, metric="confidence", min_threshold=0.6)

        if not payment_frequent_itemsets.empty:
            payment_rules = association_rules(payment_frequent_itemsets, metric="confidence", min_threshold=0.1)
            return payment_rules
        else:
            print("\n无支付方式关联规则")
            return pd.DataFrame()
    except Exception as e:
        print(f"解析错误: {e}")
        return pd.DataFrame()


# 在主流程中调用支付方式分析函数
payment_rules = analyze_payment_methods(df)
if not payment_rules.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='support', y='confidence', size='lift', data=payment_rules)
    plt.title('支付方式与商品类别的关联规则')
    plt.xlabel('支持度')
    plt.ylabel('置信度')
    plt.savefig('dog2.png', bbox_inches='tight')
    plt.show()
else:
    print("\n无支付方式关联规则可可视化")

# =============== 任务三：时间序列模式挖掘 ===============
if 'purchase_history' in df.columns:
    try:
        # 从 purchase_history 解析购买日期
        df['purchase_date'] = df['purchase_history'].apply(
            lambda x: pd.to_datetime(ast.literal_eval(x).get('purchase_date')))

        # 按时间维度分组分析
        df['quarter'] = df['purchase_date'].dt.quarter
        df['month'] = df['purchase_date'].dt.month
        df['weekday'] = df['purchase_date'].dt.day_name()

        # 分析季度购买趋势
        quarterly_purchase = df.explode('categories').groupby(
            ['categories', 'quarter']).size().unstack().fillna(0)

        # 可视化季度趋势
        plt.figure(figsize=(14, 7))
        quarterly_purchase.T.plot(kind='area', stacked=True, colormap='viridis')
        plt.title('商品类别季度购买趋势')
        plt.xlabel('季度')
        plt.ylabel('购买次数')
        plt.legend(title='商品类别', bbox_to_anchor=(1.05, 1))
        plt.savefig('seasonal_trend.png', bbox_inches='tight')
        plt.show()

        # 分析月度热门品类
        monthly_top = df.explode('categories').groupby(
            ['month', 'categories']).size().groupby(level=0).nlargest(3)
        print("\n月度热门商品类别TOP3:\n", monthly_top)

        # 简单时序模式分析（示例：前序购买关联）
        # 假设数据包含用户ID和订单序列（需根据实际数据结构调整）
        if 'user_id' in df.columns:
            from mlxtend.preprocessing import SequenceGenerator

            # 生成用户购买序列
            user_sequences = df.groupby('user_id')['categories'].apply(list)
            # 使用PrefixSpan算法（需安装prefixspan）
            # !pip install prefixspan
            from prefixspan import PrefixSpan

            ps = PrefixSpan(user_sequences)
            ps.minlen = 2
            frequent_sequences = ps.frequent(3)  # 支持度≥3次
            print("\n常见购买序列模式:\n", frequent_sequences[:10])

    except Exception as e:
        print(f"时间序列分析错误: {e}")
else:
    print("\n数据中缺少 purchase_date 字段，无法进行时间序列分析")
# =============== 任务四：退款模式分析（适配当前数据结构） ===============
try:
    # 解析关键字段
    def parse_refund_data(row):
        try:
            history = ast.literal_eval(row['purchase_history'])
            return {
                'payment_status': history.get('payment_status', '已支付'),
                'categories': [history.get('categories', '')],  # 转换为列表形式
                'avg_price': history.get('avg_price', 0)
            }
        except:
            return {
                'payment_status': '未知',
                'categories': ['未知'],
                'avg_price': 0
            }


    # 应用解析函数
    parsed_data = df.apply(parse_refund_data, axis=1)
    refund_df = pd.DataFrame(parsed_data.tolist())

    # 筛选退款数据（包含部分退款）
    refund_records = refund_df[refund_df['payment_status'].isin(['已退款', '部分退款'])]

    if not refund_records.empty:
        print(f"\n发现退款记录：{len(refund_records)}条")

        # 数据预处理
        # 展开类别列表并过滤空值
        refund_records['categories'] = refund_records['categories'].apply(
            lambda x: x if isinstance(x, list) else [x])
        refund_exploded = refund_records.explode('categories')

        # 生成独热编码矩阵
        refund_dummies = pd.get_dummies(refund_exploded['categories']).groupby(level=0).max()

        # 挖掘退款关联规则
        refund_frequent = fpgrowth(refund_dummies, min_support=0.005, use_colnames=True)

        if not refund_frequent.empty:
            # 生成关联规则
            refund_rules = association_rules(refund_frequent,
                                             metric="confidence",
                                             min_threshold=0.4)

            # 格式优化：转换frozenset为字符串
            refund_rules['antecedents'] = refund_rules['antecedents'].apply(
                lambda x: ', '.join(list(x)))
            refund_rules['consequents'] = refund_rules['consequents'].apply(
                lambda x: ', '.join(list(x)))

            # 输出TOP5有效规则
            print("\n[重要] 退款关联规则TOP5（按提升度排序）：")
            print(refund_rules.sort_values('lift', ascending=False)[[
                'antecedents', 'consequents', 'support', 'confidence', 'lift'
            ]].head(5))

            # 可视化
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=refund_rules,
                            x='support',
                            y='confidence',
                            size='lift',
                            hue='lift',
                            palette='viridis')
            plt.title('退款商品关联规则分布\n(气泡大小表示提升度)')
            plt.xlabel('支持度')
            plt.ylabel('置信度')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.savefig('refund_rules.png', bbox_inches='tight')
            plt.show()

            # 高发退款类别分析
            top_refund_cats = refund_exploded['categories'].value_counts().head(5)
            print("\n[高频] 退款率TOP5商品类别：")
            print(top_refund_cats)

            # 价格与退款关联分析
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=refund_df,
                        x='payment_status',
                        y='avg_price',
                        showfliers=False)
            plt.title('不同退款状态的商品价格分布')
            plt.savefig('price_refund.png')
            plt.show()

        else:
            print("\n未发现显著的退款关联模式")
    else:
        print("\n当前数据集没有退款记录")

except Exception as e:
    print(f"退款分析失败，错误信息：{str(e)}")

# ### 任务3：实现时间序列模式挖掘（取消注释并增强分析）
# # 转换日期字段
# df['purchase_date'] = pd.to_datetime(df['purchase_history'].apply(
#     lambda x: ast.literal_eval(x).get('purchase_date')))  # 从历史记录提取日期

# # 时间维度分析
# if 'purchase_date' in df.columns:
#     df['quarter'] = df['purchase_date'].dt.quarter
#     df['month'] = df['purchase_date'].dt.month
#     df['weekday'] = df['purchase_date'].dt.weekday_name

#     # 按季度分析购买频率
#     seasonal_pattern = df.explode('categories').groupby(
#         ['categories','quarter']).size().unstack().fillna(0)

#     # 可视化季节模式
#     plt.figure(figsize=(12,6))
#     seasonal_pattern.T.plot(kind='area', stacked=True)
#     plt.title('商品类别季度购买趋势')
#     plt.savefig('seasonal_trend.png')

# ### 任务4：完善退款模式分析（取消注释并调整参数）
# # 筛选退款数据
# refund_df = df[df['purchase_history'].apply(
#     lambda x: ast.literal_eval(x).get('payment_status') in ['已退款','部分退款'])].copy()

# if not refund_df.empty:
#     # 生成退款商品矩阵
#     refund_dummies = pd.get_dummies(refund_df['categories'].explode()).groupby(level=0).max()

#     # 调整到任务要求的参数
#     refund_frequent = apriori(refund_dummies, min_support=0.005, use_colnames=True)

#     if not refund_frequent.empty:
#         refund_rules = association_rules(refund_frequent, metric="confidence", min_threshold=0.4)
#         print("\n退款关联规则TOP5:")
#         print(refund_rules[['antecedents','consequents','support','confidence']].head())

#         # 可视化退款规则
#         plt.figure(figsize=(10,6))
#         sns.scatterplot(x='support', y='confidence', data=refund_rules)
#         plt.title('退款商品关联规则')
#         plt.savefig('refund_rules.png')
