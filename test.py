# from recommend import ImprovedCoveringBaseCollaborativeFilteringRecommend as icbcfRecommend
# from prediction import ImprovedCoveringBaseCollaborativeFilteringPredict as icbcfPrediction
# from caserec.utils.process_data import ReadFile, WriteFile
# from caserec.recommenders.rating_prediction.itemknn import ItemKNN
# from caserec.recommenders.rating_prediction.userknn import UserKNN
# from caserec.recommenders.rating_prediction.svdplusplus import SVDPlusPlus
# from ProcessData import ProcessData
#
# from Statistic import Statistic
#
# import matplotlib.pyplot as plt
# train_file = "data/u.train"
# test_file = "data/u.test"
# output_file = "output.dat"
#
# in_file = "ml-100k/u.data"
# out_train_file = "data/u.train"
# out_test_file = "data/u.test"
#
# # icbcfPrediction(train_file=train_file,test_file=test_file,cold_start_user_threshold=30,output_file=output_file,k_neighbors=60,ratio_threshold=0.9).compute()
# # UserKNN(train_file=train_file,test_file=test_file,output_file=output_file,k_neighbors=20).compute()
#
# # icbcfRecommend(train_file=train_file,test_file=test_file,cold_start_user_threshold=20,output_file=output_file).compute()
#
#
# # ProcessData(in_file,out_test_file,out_train_file,cold_start_threshold=30).export_data()
#
#
# # Statistic(in_file).propotion_items_ratings()
#
# SVDPlusPlus(train_file,test_file,factors=10).compute()
import functools
import itertools

# print({{1,2},3} | {1,2,3})
a = {"a":[1,2,3,4]}

a["a"].append(5)
print(a)