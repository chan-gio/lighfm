import json
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM

def read_json_data(products_file, sales_file):
    with open(products_file, 'r', encoding='utf-8') as file:
        products = json.load(file)
    with open(sales_file, 'r', encoding='utf-8') as file:
        sales = json.load(file)
    return products, sales

def read_user_data(user_file):
    with open(user_file, 'r', encoding='utf-8') as file:
        users = json.load(file)
    user_ids = [user['user_id'] for user in users['users']]
    return user_ids

def parse_product_data(data):
    product_categories = {}
    products = data.get('products', [])

    for product in products:
        if 'product_id' in product and 'category' in product:
            product_id = product['product_id']
            category = product['category']
            product_categories[product_id] = category   
    return product_categories

def parse_product_names(data):
    product_names = {}
    products = data.get('products', [])

    for product in products:
        if 'product_id' in product and 'product_name' in product:
            product_id = product['product_id']
            product_name = product['product_name']
            product_names[product_id] = product_name
    return product_names

def parse_sales_data(data):
    interactions = []
    sales = data.get('sales', [])
    for sale in sales:
        if 'customer' in sale and 'product_id' in sale and 'quantity' in sale:
            customer_id = str(sale['customer'].get('customer_id', None))
            product_id = sale['product_id']
            quantity = int(sale['quantity'])
            if customer_id is not None:
                interactions.append((customer_id, product_id, quantity))
            else:
                print("Warning: Missing 'customer_id' in sale data:", sale)
        else:
            print("Warning: Missing required fields in sale data:", sale)
    return interactions

def build_interaction_matrix(interactions, product_categories):
    # Tạo một tập hợp chứa tất cả các customer_id và product_id
    customers = {customer_id for customer_id, _, _ in interactions}
    products = {product_id for _, product_id, _ in interactions}
    num_customers = len(customers)
    num_products = len(products)
    
    # Tạo mapping từ customer_id và product_id sang index
    customer_id_map = {customer_id: i for i, customer_id in enumerate(customers)}
    product_id_map = {product_id.upper(): i for i, product_id in enumerate(products)}
    
    # Tạo danh sách các hàng, cột và dữ liệu cho ma trận
    rows = []
    cols = []
    data = []

    for customer_id, product_id, quantity in interactions:
        # Lấy index của customer và product từ mapping
        customer_index = customer_id_map[customer_id]
        product_index = product_id_map[product_id]
        # Thêm dữ liệu vào danh sách
        rows.append(customer_index)
        cols.append(product_index)
        data.append(quantity)

    # Tạo ma trận tương tác từ danh sách hàng, cột và dữ liệu
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(num_customers, num_products), dtype=np.int32)
    # Trả về ma trận tương tác, mapping từ product_id sang index, mapping từ customer_id sang index, số lượng khách hàng và số lượng sản phẩm
    return interaction_matrix, product_id_map, customer_id_map, num_customers, num_products

# Đọc dữ liệu từ các file JSON
products, sales = read_json_data('products.json', 'sales.json')
user_ids = read_user_data('user.json')

# Parse dữ liệu sản phẩm và xây dựng mapping từ product_id sang category
product_categories = parse_product_data(products)
product_names = parse_product_names(products)

# Parse dữ liệu sales và xây dựng ma trận tương tác
interactions = parse_sales_data(sales)
interaction_matrix, product_id_map, customer_id_map, num_customers, num_products = build_interaction_matrix(interactions, product_categories)

# Tạo mô hình
model = LightFM(loss='warp')
# Huấn luyện mô hình
model.fit(interaction_matrix, epochs=30, num_threads=2)

def sample_recommendation_same_category(model, interaction_matrix, customer_ids, customer_id_map, product_id_map, product_categories, product_names):
    num_customers, num_products = interaction_matrix.shape

    for customer_id in customer_ids:
        if customer_id in customer_id_map:
            customer_index = customer_id_map[customer_id]
            if 0 <= customer_index < interaction_matrix.shape[0]:
                known_positives_indices = interaction_matrix[customer_index].nonzero()[1]
                known_positives_categories = [product_categories.get(list(product_id_map.keys())[product_id]) for product_id in known_positives_indices]

                scores = model.predict(customer_index, np.arange(num_products))
                top_items_indices = np.argsort(-scores)
                top_items = [list(product_id_map.keys())[product_id] for product_id in top_items_indices]
                top_items_categories = [product_categories.get(item) for item in top_items]

                print("Customer %s" % customer_id)
                print(" Known positives:")
                for i in range(min(3, len(known_positives_categories))):
                    product_id = list(product_id_map.keys())[known_positives_indices[i]]
                    product_name = product_names.get(product_id)
                    print("     %s (Category: %s)" % (product_name, known_positives_categories[i]))

                print(" Recommended:")
                recommended_products = []
                for item, category in zip(top_items, top_items_categories):
                    if category in known_positives_categories:
                        product_name = product_names.get(item)
                        print("     %s (%s) (Category: %s)" % (product_name, item, category))
                        recommended_products.append(category)
                        if len(recommended_products) == 3:
                            break

def recommend_for_all_customers(model, interaction_matrix, user_ids, customer_id_map, product_id_map, product_categories, product_names):
    for customer_id in user_ids:
        sample_recommendation_same_category(model, interaction_matrix, [customer_id], customer_id_map, product_id_map, product_categories, product_names)

# Gọi hàm để đề xuất sản phẩm cho người dùng
recommend_for_all_customers(model, interaction_matrix, user_ids, customer_id_map, product_id_map, product_categories, product_names)
