from caserec.utils.process_data import ReadFile
from caserec.utils.process_data import WriteFile
import random


class ProcessData(object):
    def __init__(self, input_file, out_test_file, out_train_file, cold_start_threshold=20, seperate=0.2):
        self.input_file = input_file
        self.dataset = ReadFile(self.input_file).read()
        self.cold_start_threshold = cold_start_threshold
        self.cold_start_user = self.find_cold_start_user()
        self.train_data = []
        self.test_data = []
        self.seperate = seperate
        self.out_test_file = out_test_file
        self.out_train_file = out_train_file

    def find_cold_start_user(self):
        cold_start_user = []
        for u in self.dataset["items_seen_by_user"]:
            if len(self.dataset["items_seen_by_user"][u]) < self.cold_start_threshold:
                cold_start_user.append(u)
        return cold_start_user

    def process_data(self):
        self.train_data = []
        self.test_data = []

        for u in self.dataset["users"]:
            if u in self.cold_start_user:
                # Chọn ngẫu nhiên 20% mục đã đánh giá
                test_item = random.choices(list(self.dataset["items_seen_by_user"][u]),
                                           k=int(self.cold_start_threshold * self.seperate))
                for i in self.dataset["feedback"][u]:
                    if i in test_item:
                        self.test_data.append([u, i, self.dataset["feedback"][u][i]])
                    else:
                        self.train_data.append([u, i, self.dataset["feedback"][u][i]])
            else:
                for i in self.dataset["feedback"][u]:
                    self.train_data.append([u, i, self.dataset["feedback"][u][i]])

    def export_data(self):
        self.process_data()
        WriteFile(self.out_train_file, self.train_data).write()
        WriteFile(self.out_test_file, self.test_data).write()
        print("Done!")
