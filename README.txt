Lớp ImprovedCoveringBaseCollaborativeFilteringPredict kế thừa lớp UserKNN của framework Case Recommender thực hiện dự đoán rating_score. Lớp này chỉ phục vụ việc đánh giá.
Lớp ImprovedCoveringBaseCollaborativeFilteringRecommend kế thừa lớp UserKNN của framework Case Recommender thực hiện gợi ý top N phim cho người dùng mới. Lớp này là lớp chính của mô hình.
Lớp ProcessData thực hiện tiền xử lý dữ liệu, tạo ra tập dữ liệu huấn luyện và dữ liệu kiểm thử.
Lớp Statistic thực hiện thống kê dữ liệu.
u.test, u.train lần lượt là tập dữ liệu kiểm thử và tập dữ liệu huấn luyện được tạo ra từ tập dữ liệu MovieLens 100K.
