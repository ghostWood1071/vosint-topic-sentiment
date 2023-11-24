# Các bước sử dụng Phân tích cảm xúc Version 2

- Hướng dẫn cài đặt các thư viện để chạy được mô hình Phân tích cảm xúc Version 2

# 1. Clone Project

- **Chạy câu lệnh:** git clone https://gitlab.aiacademy.edu.vn/research-develop/nlp/v_osint_topic_sentiment.git
- **Checkout sang branch "version2":** git checkout version_2
# 2. Hướng dẫn Download Mô Hình và cài đặt các thư viện cần thiết

## 2.1. Cài thư viện:

- **Chạy câu lệnh:** pip install -r ./requirements.txt

## 2.2. Download mô hình:

- **Chạy câu lệnh:** cd v_osint_topic_sentiment
- **Chạy câu lệnh:** python download_model.py

## 3. Hướng dẫn sử dụng thư viện

    - File gọi thư viện cùng cấp với File api_main.py

## 3.1. Đối với các bản tin:

    from v_osint_topic_sentiment.sentiment_analysis import topic_sentiment_classification

    title = "Nước về hồ thủy điện tăng nhanh, vơi nỗi lo cắt điện"

    descriptiion = ""

    content = "Thông tin trên được Cục Kỹ thuật an toàn và Môi trường công nghiệp (Bộ Công Thương) cập nhật trong báo cáo ngày 27/6. Theo cơ quan này, lưu lượng về các hồ chứa thủy điện khu vực Bắc Bộ, Đông Nam Bộ, Tây Nguyên giảm nhẹ; khu vực Bắc Trung Bộ tăng nhẹ so với hôm qua; khu vực duyên hải Nam Trung Bộ lưu lượng về hồ thấp, dao động nhẹ so với ngày hôm qua. Lưu lượng về các hồ chứa ở các tỉnh Hà Giang, Bắc Kạn, Lạng Sơn đã đạt đỉnh và giảm dần. Mực nước hồ chứa thủy điện khu vực Bắc Bộ tăng nhanh, cao hơn mực nước chết từ 7-20m; khu vực Bắc Trung Bộ, Đông Nam Bộ tăng nhẹ; khu vực Tây Nguyên, duyên hải Nam Trung Bộ giảm nhẹ so với ngày hôm qua, mực nước các hồ nằm trong phạm vi mực nước tối thiểu theo quy định của Quy trình vận hành. Lưu lượng, mực nước tại các hồ thủy điện khu vực Bắc Bộ tăng, các hồ chứa lớn đang nâng cao mực nước, hạn chế huy động phát điện để dự phòng cho đợt nắng nóng tiếp theo, một số hồ vừa, nhỏ, tràn tự do đã phải điều tiết nước lũ. Các hồ thủy điện lưu lượng về hồ lớn, giảm nhẹ so với ngày hôm qua. Cụ thể: Hồ Lai Châu: 195 m3/s; Hồ Sơn La: 561 m3/s; Hồ Hòa Bình: 773 m3/s; Hồ Thác Bà: 114 m3/s; Hồ Tuyên Quang: 396 m3/s; Hồ Bản Chát: 210,5 m3/s. Mực nước các hồ tăng cao so với ngày hôm qua, mực nước hồ/mực nước chết hiện như sau: Hồ Lai Châu 290.21 m/265 m; Hồ Sơn La 182.97/175 m; Hồ Hòa Bình 102,23/80m; Hồ Thác Bà 47,25/46 m; Hồ Tuyên Quang 102,61/90m; Hồ Bản Chát 445,09/431m. Lượng nước về một số hồ khu vực Bắc Trung Bộ, Đông Nam Bộ thấp, chủ yếu điều tiết nước đảm bảo dòng chảy tối thiểu, phát điện cầm chừng để đảm bảo an toàn tổ máy khi vận hành, nâng cao mực nước phát điện. Các hồ mực nước thấp, gồm: Thác Bà, Bản Vẽ, Đồng Nai 3, Thác Mơ. Một số thủy điện phát điện hạn chế, cầm chừng với lưu lượng, mực nước, công suất thấp, như: Thác Bà, Bản Vẽ, Thác Mơ, Đồng Nai 3. Dự báo tình hình thủy văn, lưu lượng nước về hồ 24h tới các hồ khu vực Bắc Bộ giảm nhẹ ở mức cao; khu vực Bắc Trung Bộ, Tây Nguyên, Đông Nam Bộ tăng nhẹ; khu vực duyện hải Nam Trung Bộ giảm nhẹ, ở mức thấp. Thủy điện phía Bắc thoát mực nước chết 7-20m, vẫn 'ăn dè' lo đợt nắng nóng tớiBáo cáo cập nhật tình hình hồ thủy điện của Cục Kỹ thuật an toàn và Môi trường công nghiệp (Bộ Công Thương) cho hay, trong ngày 26/6, mực nước hồ chứa thủy điện khu vực Bắc Bộ tăng nhanh, cao hơn mực nước chết từ 7-20m."

    predict_out = topic_sentiment_classification(title=title,description=descriptiion,content=content)

    print(predict_out)

    {'sentiment_label': 'trungtinh'}

## 3.2. Đối với FaceBook:

    from v_osint_topic_sentiment.sentiment_analysis import topic_sentiment_classification

    face_text = """ĐTQG nữ Việt Nam lên đường sang New Zealand tham dự World Cup 2023.

    Chúc may mắn các cô gái Việt Nam"""

    predict_out = topic_sentiment_classification(title="",description="",content=face_text)

    print(predict_out)

    {'sentiment_label': 'tich_cuc'}
