import functools
import itertools
import sys

from caserec.recommenders.item_recommendation.base_item_recommendation import BaseItemRecommendation


class ReductFindingRecommend(BaseItemRecommendation):
    def __init__(self, train_file, test_file, output_file=None, as_binary=False, rank_length=10,
                 similarity_metric="cosine", sep='\t', output_sep='\t'):
        self.dic_reducts = {}
        sys.setrecursionlimit(10**6)
        super().__init__(train_file, test_file, output_file, as_binary, rank_length, similarity_metric, sep, output_sep)

    # override
    # định nghĩa thêm vài giá trị trong train_set
    def read_files(self):
        super().read_files()

        self.train_set["users_not_viewed_items"] = {}
        self.train_set["items_not_seen_by_user"] = {}

        for item, set_users in self.train_set["users_viewed_item"].items():
            self.train_set["users_not_viewed_items"].setdefault(item, set(set(self.train_set["users"]) - set_users))

        for user, set_items in self.train_set["items_seen_by_user"].items():
            self.train_set["items_not_seen_by_user"].setdefault(user, set(set(self.train_set["items"]) - set_items))

    # Bắt đầu thuật toán reduct finding
    def process_reduct_finding(self):
        for user in self.train_set["users"]:
            cl = set(self.train_set["items_seen_by_user"][user])
            dl = set(self.train_set["items"]) - cl

            self.dic_reducts.setdefault(user, [cl])

            dependencyCL = self.calculateDependency(cl, dl)

            self.reduct_finding(cl, cl, user, dl, dependencyCL)
            print(self.dic_reducts)



    def reduct_item_for_user(self, userData):
        coverS = []

        for i in range(1, len(userData)):
            # List các tập hơp có i phần tử
            combination = list(itertools.combinations(userData, i))

            for item in combination:
                cover = self.calculateCoverS(set(item))
                if cover not in coverS and len(cover) > 0:
                    coverS += (value for value in cover if value not in coverS)

        return coverS

    # Hàm tính phủ cảm sinh
    def calculateCoverS(self, set_items):
        coverS = []
        for i in range(0, len(set_items)):
            combination = list(itertools.combinations(set_items, i + 1))

            for element in combination:
                remain_items = set(set_items) - set(element)

                cover = [user for (user, items) in self.train_set["items_seen_by_user"].items()
                         if
                         set(element).issubset(items) and set(remain_items) & items == set()]

                if cover not in coverS and len(cover) > 0:
                    coverS.append(cover)

        set_trans_not_include_all = [user for (user, items) in self.train_set["items_seen_by_user"].items()
                                     if
                                     set(set_items) & items == set()]

        if set_trans_not_include_all not in coverS and len(set_trans_not_include_all) > 0:
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
            dependencyCCL = self.calculateDependency(ccl, dl)

            if dependencyCCL == dependencyCL:
                self.dic_reducts[user].append(ccl)

                if pccl in self.dic_reducts[user]:
                    self.dic_reducts[user].remove(pccl)

                for item in sccl:
                    self.reduct_finding(item, ccl, user, dl, dependencyCL)

    # Hàm sinh ra tất cả các tập con từ một tập cha
    def generateAllChid(self, parent):
        childs = []
        for i in range(len(parent) - 1):
            childs += itertools.combinations(parent, i + 1)

        return childs

    # Hàm tính độ phụ thuộc
    def calculateDependency(self, cl, dl):
        covCLu = {}
        covDLu = {}
        topCL = []

        cover_cl = self.calculateCoverS(cl)
        cover_dl = self.calculateCoverS(dl)

        for i in range(len(cl)):
            topCL += itertools.combinations(cl, i + 1)

        # for each user tìm covClu và covDLu dựa trên cover_cl và cover_dl
        for user in self.train_set["users"]:
            covCLu.setdefault(user, set.intersection(*[set(x) for x in cover_cl if user in x]))
            covDLu.setdefault(user, set.intersection(*[set(x) for x in cover_dl if user in x]))

        posCL = []
        for (key, value) in covCLu.items():
            for (keyDL, valueDL) in covDLu.items():
                join = set(value) & set(valueDL)

                if join not in posCL and join != set():
                    posCL.append(join)

        pCL = len(posCL) / len(topCL)

        return pCL