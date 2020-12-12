from builtins import str

from caserec.recommenders.rating_prediction.itemknn import ItemKNN
from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.utils.extra_functions import print_header
from caserec.utils.extra_functions import timed
from scipy.spatial.distance import squareform, pdist
import numpy as np
import os
from collections import OrderedDict


class ImprovedCoveringBaseCollaborativeFilteringPredict(UserKNN):

    # Hàm khởi tạo
    def __init__(self, train_file, test_file, output_file=None, as_binary=False, rank_length=10,
                 similarity_metric="cosine", sep='\t', output_sep='\t', ratio_threshold=0.9, k_neighbors=10,
                 cold_start_user_threshold=5):
        self.ratio_threshold = ratio_threshold
        self.cold_start_user_threshold = cold_start_user_threshold

        self.cold_start_users = []

        self.reduct_users = []

        self.popular_items = []

        self.nu_knn = {}

        super().__init__(train_file, test_file, output_file=output_file, k_neighbors=k_neighbors)

    def calculateColdStartUser(self):
        items_seen_by_user = self.train_set["items_seen_by_user"]
        self.cold_start_users = []
        for u in items_seen_by_user:
            if len(items_seen_by_user[u]) <= self.cold_start_user_threshold:
                self.cold_start_users.append(u)

    # Thuật toán Covering redution
    # Trả ra danh sách user đã bị reduce
    def coveringRedution(self):
        result = []
        for user in self.train_set["users"]:
            for user2 in self.train_set["users"]:
                if self.isSubset(list(self.train_set["items_seen_by_user"][user]),
                                 list(self.train_set["items_seen_by_user"][user2])) and user2 != user:
                    if not user2 in result:
                        result.append(user2)
        return result

    # Hàm kiểm tra b có phải là tập con của a ko
    def isSubset(self, a, b):
        if len(a) < len(b):
            return False
        else:
            for item in b:
                if not item in a:
                    return False

            return True

    # Hàm xây dựng lớp quyết định
    # Trả ra danh sách item bị reduce
    def CalculateDescisionClass(self):
        len_items = len(self.train_set["items"])
        items_seen_by_user = self.train_set["users_viewed_item"]

        orderList = sorted(items_seen_by_user, key=lambda k: len(items_seen_by_user[k]), reverse=True)

        while (len_items - len(self.popular_items)) / len_items >= self.ratio_threshold:
            items_remove = orderList.pop(0)
            if items_remove in self.train_set["items"]:
                self.train_set["items"].remove(items_remove)
            if items_remove in self.test_set["items"]:
                self.test_set["items"].remove(items_remove)

            for u in self.train_set["items_seen_by_user"]:
                if items_remove in self.train_set["items_seen_by_user"][u]:
                    self.train_set["items_seen_by_user"][u].remove(items_remove)
            for u in self.test_set["items_seen_by_user"]:
                if items_remove in self.test_set["items_seen_by_user"][u]:
                    self.test_set["items_seen_by_user"][u].remove(items_remove)

            if items_remove in self.train_set["users_viewed_item"]:
                self.train_set["users_viewed_item"].pop(items_remove)
            if items_remove in self.test_set["users_viewed_item"]:
                self.test_set["users_viewed_item"].pop(items_remove)

            for u in self.train_set["feedback"]:
                if items_remove in self.train_set["feedback"][u]:
                    self.train_set["feedback"][u].pop(items_remove)
            for u in self.test_set["feedback"]:
                if items_remove in self.test_set["feedback"][u]:
                    self.test_set["feedback"][u].pop(items_remove)

            self.popular_items.append(items_remove)

        print("popular_items" + str(len(self.popular_items)))

    def CalculateReductUsers(self):
        self.reduct_users = self.coveringRedution()

        for user in self.reduct_users:
            if user in self.cold_start_users:
                self.reduct_users.remove(user)

        for u in self.reduct_users:
            if u in self.train_set["users"]:
                self.train_set["users"].remove(u)

            for i in self.train_set["users_viewed_item"]:
                if u in self.train_set["users_viewed_item"][i]:
                    self.train_set["users_viewed_item"][i].remove(u)

            if u in self.train_set["items_seen_by_user"]:
                self.train_set["items_seen_by_user"].pop(u)

            if u in self.train_set["feedback"]:
                self.train_set["feedback"].pop(u)

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t',
                n_ranks=None):
        # read files
        self.read_files()

        self.calculateColdStartUser()

        self.CalculateDescisionClass()

        self.CalculateReductUsers()

        # initialize empty predictions (Don't remove: important to Cross Validation)
        self.predictions = []

        if verbose:
            test_info = None

            main_info = {
                'title': 'Rating Prediction > ' + self.recommender_name,
                'n_users': len(self.train_set['users']),
                'n_items': len(self.train_set['items']),
                'n_interactions': self.train_set['number_interactions'],
                'sparsity': self.train_set['sparsity']
            }

            if self.test_file is not None:
                test_info = {
                    'n_users': len(self.test_set['users']),
                    'n_items': len(self.test_set['items']),
                    'n_interactions': self.test_set['number_interactions'],
                    'sparsity': self.test_set['sparsity']
                }

            print_header(main_info, test_info)

            self.init_model()
            print("training_time:: %4f sec" % timed(self.train_baselines))
            if self.extra_info_header is not None:
                print(self.extra_info_header)
            print("prediction_time:: %4f sec" % timed(self.predict))

        else:
            # Execute all in silence without prints
            self.extra_info_header = None
            self.init_model()
            self.train_baselines()
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)

    def remove_user_not_cold_start_test_set(self):
        for u in self.test_set["users"]:
            if u not in self.cold_start_users:
                self.test_set["users"].remove(u)

                # for i in self.test_set["users_viewed_item"]:
                #     if u in self.test_set["users_viewed_item"][i]:
                #         self.test_set["users_viewed_item"][i].remove(u)
                #
                # if u in self.test_set["items_seen_by_user"]:
                #     self.test_set["items_seen_by_user"].pop(u)

                if u in self.test_set["feedback"]:
                    self.test_set["feedback"].pop(u)

        with open('ml-100k/cbcf_data.test', 'w') as infile:
            for u in self.test_set['feedback']:
                for i in self.test_set['feedback'][u]:
                    infile.write(str(u) + "\t" + str(i) + "\t" + str(self.test_set['feedback'][u][i]) + "\n")
