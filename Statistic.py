from caserec.utils.process_data import ReadFile
import matplotlib.pyplot as pylot


class Statistic:
    def __init__(self, input_file):
        self.input_file = input_file
        self.dataset = ReadFile(self.input_file).read()

    def number_user_per_score(self):
        count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for u in self.dataset["feedback"]:
            for i in self.dataset["feedback"][u]:
                count[int(self.dataset["feedback"][u][i])] += 1

        names = [1, 2, 3, 4, 5]
        values = [count[1], count[2], count[3], count[4], count[5]]

        pylot.bar(names, values)
        pylot.title("Số lượng đánh giá trên mỗi giá trị rating_score")
        pylot.show()

    def propotion_items_ratings(self):
    #     # print(list(filter(lambda x: len(x[1]) >= 10000, self.dataset["users_viewed_item"])))
    #
        greater100 = list((i for i in self.dataset["users_viewed_item"] if len(self.dataset["users_viewed_item"][i])<100))
        greater100_number = len(greater100)
        print(greater100_number)
        greater100_rating = len(greater100) / len(self.dataset["items"])
        print(greater100_rating)

        rating_number = 0
        for i in greater100:
            rating_number+= len(self.dataset["users_viewed_item"][i])

        print(rating_number)
    #
    #     greater50 = list((i for i in self.dataset["users_viewed_item"] if len(self.dataset["users_viewed_item"][i])>50 and len(self.dataset["users_viewed_item"][i]) < 100))
    #     greater50_number = len(greater50)
    #     greater50_rating = len(greater50) / len(self.dataset["items"])
    #
    #     greater10 = list((i for i in self.dataset["users_viewed_item"] if len(self.dataset["users_viewed_item"][i])>50 and len(self.dataset["users_viewed_item"][i]) < 100))
    #     greater10_number = len(greater1k)
    #     greater10_rating = len(greater5k) / len(self.dataset["items"])
    #
    #     less1k = filter(lambda x: len(x) < 1000, self.dataset["users_viewed_item"])
    #     less1k_number = len(less1k)
    #     less1k_rating = len(less1k) / len(self.dataset["items"])
    #
    #     data = [
    #         [greater100_number, greater100_rating],
    #         [greater5k_number, greater5k_rating],
    #         [greater1k_number, greater1k_rating],
    #         [less1k_number,less1k_rating]
    #     ]
    #
    #     columns = ["Item Number","Item rate (%)"]
    #     rows = ["Ratings ≥ 10K","5K ≤ ratings < 10K","1K ≤ ratings < 5K","Ratings ≤ 1K"]
    #
    #     pylot.table(cellText=data)


