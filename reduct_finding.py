from caserec.recommenders.item_recommendation.base_item_recommendation import BaseItemRecommendation
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.utils.extra_functions import print_header
from caserec.utils.extra_functions import timed
from scipy.spatial.distance import squareform, pdist
import numpy as np
import os
from collections import OrderedDict
import itertools
import functools


class ReductFindingRecommend(UserKNN):
    def __init__(self):
        self.dic_reducts = {}

    # override
    # định nghĩa thêm vài giá trị trong train_set
    def read_files(self):
        super()

        self.train_set["users_not_viewed_items"] = {}
        self.train_set["items_seen_by_user"] = {}

        for item in self.train_set["users_viewed_items"]:
            self.train_set["users_not_viewed_items"].setdefault(item, set()).add(
                set(self.train_set["users"]).difference(self.train_set["users_viewed_items"][item]))

        for user in self.train_set["items_seen_by_user"]:
            self.train_set["items_seen_by_user"].setdefault(user, set()).add(
                set(self.train_set["items"]).difference(self.train_set["items_seen_by_user"][user]))

    # Bắt đầu thuật toán reduct finding
    def process_reduct_finding(self):
        for user in self.train_set["users"]:
            self.dic_reducts.setdefault(user, list())

            cl = set(self.train_set["items_seen_by_user"][user])
            dl = set(self.train_set["items"]) - cl

            dependencyCL = self.calculateDependency(cl, dl, user)

            self.reduct_finding(cl, cl, user, dl, dependencyCL)

    def reduct_item_for_user(self, userData):
        coverS = []

        for i in range(1, len(userData)):
            # List các tập hơp có i phần tử
            combination = list(itertools.combinations(userData, i))

            for item in combination:
                coverS.append(self.calculateCoverS(item))

        return coverS

    # Hàm tính phủ cảm sinh
    def calculateCoverS(self, items):
        coverS = []
        for i in range(1, len(items)):
            combination = list(itertools.combinations(items, i))

            for items in combination:
                remain_items = items.symmetric_difference(items)
                set_trans_include_items = {functools.reduce(lambda a, b: set(a) & set(b),
                                                            [v for (k, v) in self.train_set["users_viewed_item"] if
                                                             k in items])}
                set_trans_not_include_remain_items = {functools.reduce(lambda a, b: set(a) & set(b),
                                                                       [v for (k, v) in
                                                                        self.train_set["users_not_viewed_item"] if
                                                                        k in remain_items])}
                intersec = set_trans_include_items & set_trans_not_include_remain_items

                if intersec not in coverS:
                    coverS.append(intersec)

        # Trường hợp giao dịch không chứa tất cả item
        set_trans_not_include_all = {functools.reduce(lambda a, b: set(a) & set(b),
                                                      [v for (k, v) in self.train_set["users_not_viewed_item"] if
                                                       k in items])}
        if set_trans_not_include_all not in coverS:
            coverS.append(set_trans_not_include_all)

        return coverS

    # Hàm sinh GCRL
    # ccl là dàn điều kiện hiện thời
    # pccl là dàn cha
    def reduct_finding(self, ccl, pccl, user, dl, dependencyCL):
        sccl = self.generateAllChid(ccl)
        if ccl == pccl:
            for item in sccl:
                self.reduct_finding(item, ccl, user, dl, dependencyCL)
        else:
            dependencyCCL = self.calculateDependency(ccl, dl, user)

            if dependencyCCL == dependencyCL:
                self.dic_reducts[user].append(ccl)

                if pccl in self.dic_reducts[user]:
                    self.dic_reducts[user].remove(pccl)

                for item in sccl:
                    self.reduct_finding(item, ccl, user, dl, dependencyCL)

    # Hàm sinh ra tất cả các tập con từ một tập cha
    def generateAllChid(self, parent):
        childs = []
        for i in range(len(parent)):
            childs += itertools.combinations(parent, i + 1)

        return childs

    # Hàm tính độ phụ thuộc
    def calculateDependency(self, cl, dl, user):
        cCL = self.reduct_item_for_user(cl)
        cDL = self.reduct_item_for_user(dl)
        topCL = []
        covCLu = {}
        covDLu = {}

        for i in range(len(cl)):
            topCL += itertools.combinations(cl, i + 1)

        covCLu.setdefault(user, functools.reduce(lambda a, b: a & b, list(filter(lambda x: user in x, cCL))))
        covDLu.setdefault(user, functools.reduce(lambda a, b: a & b, list(filter(lambda x: user in x, cDL))))

        posCL = []
        for (key, value) in covCLu:
            for (keyDL, valueDL) in covDLu:
                join = set(value) & set(valueDL)
                if join not in posCL:
                    posCL.append(join)

        pCL = len(posCL) / len(topCL)

        return pCL
