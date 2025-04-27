# Kế hoạch dự án Phân loại Biển báo Giao thông GTSRB

## Danh sách công việc (Project Tasks)

1.  **Tìm hiểu bài toán và dữ liệu (Problem & Data Understanding):**
    *   Xác định rõ mục tiêu: Phân loại chính xác ảnh biển báo giao thông Đức vào các lớp tương ứng.
    *   Nghiên cứu bộ dữ liệu GTSRB:
        *   Số lượng lớp (loại biển báo) là bao nhiêu?
        *   Định dạng dữ liệu (ảnh, file chú thích)?
        *   Cấu trúc thư mục (tập huấn luyện, kiểm tra)?
        *   Đặc điểm ảnh (kích thước, màu sắc, các yếu tố nhiễu như ánh sáng, che khuất, góc nhìn)?
        *   Kiểm tra sự mất cân bằng dữ liệu giữa các lớp.
    *   **Phân tích cấu trúc thư mục (Kết quả sơ bộ):**
        *   Thư mục gốc: `GTSRB/`
        *   Thư mục con chính: `Final_Training/` và `Final_Test/`.
        *   `Final_Training/Images/`: Chứa 43 thư mục con (`00000` đến `00042`), mỗi thư mục đại diện cho một lớp và chứa các ảnh huấn luyện tương ứng. => **Tổng cộng 43 lớp.**
        *   `Final_Test/Images/`: Chứa các ảnh kiểm tra.
        *   `Final_Test/GT-final_test.csv`: Chứa thông tin và nhãn (ground truth) cho các ảnh kiểm tra.

2.  **Tiền xử lý dữ liệu (Data Preprocessing):**
    *   Tải dữ liệu (ảnh và nhãn).
    *   Thay đổi kích thước ảnh về một kích thước thống nhất phù hợp cho mô hình.
    *   Chuẩn hóa giá trị pixel (ví dụ: về khoảng [0, 1] hoặc [-1, 1]).
    *   Tăng cường dữ liệu (Data Augmentation) - *nên thực hiện*: Tạo thêm các biến thể ảnh (xoay, dịch chuyển, phóng to/thu nhỏ, thay đổi độ sáng...) để mô hình học được các đặc trưng tổng quát hơn và chống overfitting.
    *   Phân chia dữ liệu (nếu chưa được chia sẵn): Tập huấn luyện (training), tập xác thực (validation), và tập kiểm tra (testing).

3.  **Lựa chọn mô hình (Model Selection):**
    *   Chọn kiến trúc mô hình phù hợp. Mạng Nơ-ron Tích chập (Convolutional Neural Networks - CNNs) là lựa chọn tiêu chuẩn cho bài toán phân loại ảnh.
        *   Có thể bắt đầu với một kiến trúc CNN đơn giản.
        *   Xem xét các kiến trúc nổi tiếng (ví dụ: LeNet, AlexNet, VGG, ResNet). Có thể cân nhắc sử dụng kỹ thuật học chuyển giao (Transfer Learning) với các mô hình đã được huấn luyện trước.
    *   Chọn framework/thư viện: TensorFlow/Keras hoặc PyTorch là những lựa chọn phổ biến.

4.  **Huấn luyện mô hình (Model Training):**
    *   Định nghĩa hàm mất mát (Loss Function): Ví dụ `Categorical Cross-Entropy` cho bài toán phân loại đa lớp.
    *   Chọn thuật toán tối ưu (Optimizer): Ví dụ Adam, SGD.
    *   Thiết lập các siêu tham số (Hyperparameters): Tốc độ học (learning rate), kích thước lô (batch size), số lượng kỷ nguyên (epochs).
    *   Huấn luyện mô hình trên tập dữ liệu huấn luyện.
    *   Theo dõi quá trình huấn luyện bằng tập xác thực (theo dõi loss, accuracy). Sử dụng các kỹ thuật như dừng sớm (Early Stopping) hoặc lưu lại mô hình tốt nhất (Model Checkpointing).

5.  **Đánh giá mô hình (Model Evaluation):**
    *   Đánh giá mô hình đã huấn luyện trên tập kiểm tra (dữ liệu chưa từng thấy).
    *   Sử dụng các độ đo: Độ chính xác (Accuracy), Precision, Recall, F1-score, Ma trận nhầm lẫn (Confusion Matrix).
    *   Phân tích các trường hợp mô hình dự đoán sai để hiểu điểm yếu.

6.  **Cải thiện mô hình (Model Improvement - Lặp lại):**
    *   Tinh chỉnh siêu tham số.
    *   Thử nghiệm các kiến trúc mô hình khác hoặc kỹ thuật tăng cường dữ liệu khác.
    *   Xử lý vấn đề mất cân bằng dữ liệu (nếu có).
    *   Thử nghiệm sâu hơn với học chuyển giao.

7.  **Triển khai (Deployment - Tùy chọn):**
    *   Lưu lại mô hình đã huấn luyện tốt nhất.
    *   Xây dựng giao diện (ví dụ: web app, API) để sử dụng mô hình phân loại ảnh mới.

## Kế hoạch nghiên cứu (Research Plan)

*   **Dataset GTSRB:** Tìm tài liệu chính thức, các bài báo khoa học đã sử dụng bộ dữ liệu này, các bước tiền xử lý phổ biến, và kết quả benchmark đã được báo cáo.
*   **Kiến trúc CNN:** Nghiên cứu các kiến trúc CNN phổ biến cho phân loại ảnh, đặc biệt là những kiến trúc đã được áp dụng thành công cho bài toán nhận diện biển báo giao thông.
*   **Kỹ thuật tăng cường dữ liệu:** Tìm hiểu các phương pháp tăng cường dữ liệu hiệu quả cho ảnh biển báo giao thông.
*   **Tài liệu Framework:** Tham khảo tài liệu của TensorFlow/Keras hoặc PyTorch để biết chi tiết về cách triển khai.
*   **Độ đo đánh giá:** Hiểu rõ ý nghĩa và cách tính toán các độ đo đánh giá hiệu năng mô hình. 