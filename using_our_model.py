
import pickle
import pandas as pd
from main import ModelWrapper


test_review_path = "devided_dataset_v2/CDs_and_Vinyl/test2/review_test.json"
test_product_path = "devided_dataset_v2/CDs_and_Vinyl/test2/product_test.json"
test_product_data = pd.read_json(test_product_path)
test_review_data = pd.read_json(test_review_path)
test_review_data["summary"] = test_review_data["summary"].fillna("negative")
test_review_data["reviewText"] = test_review_data["reviewText"].fillna("negative")
test_review_data['vote'] = test_review_data['vote'].apply(lambda x: 0 if x == None else int(x.replace(',','')))
test_review_data['image'] = test_review_data['image'].apply(lambda x: False if x == None else True)
test_review_data['summary'] = test_review_data['summary'].apply(lambda x: "bad" if x == "" else x)
test_review_data['reviewText'] = test_review_data['reviewText'].apply(lambda x: "bad" if x == "" else x)



# Load the custom object from the pickle file
with open('amazon_reviews_model.pickle', 'rb') as f:
    model_wrapper = pickle.load(f)

# Extract the model and variables from the wrapper object
model = model_wrapper.model


iDs = list(test_product_data['asin'])
feature_vector_2 = pd.DataFrame({"aw_rt":[], "naw_rt":[], "aw_s":[], "naw_s":[],"vote_score":[], "image_score":[], "verified":[], "time_score":[]})
test_feature_vector = model_wrapper.make_feature_vector(iDs, feature_vector_2, test_review_data, "testing")
predicted_class = model.predict(test_feature_vector)

final_json = test_product_data
final_json["awesomeness"] = predicted_class
final_json.to_json("predictions.json")

