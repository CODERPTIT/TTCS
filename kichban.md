# Kịch bản Thuyết trình Hội đồng: Kiến trúc KAN-Attention U-Net 3D

*Lưu ý cho người thuyết trình: Kịch bản được chia làm 2 phần: [Nội dung trên Slide] (ngắn gọn, trực quan) và [Lời thoại] (phần bạn sẽ nói). Hãy nói với tốc độ vừa phải, nhấn mạnh vào các từ in đậm.*

---

## Slide 1: Tiêu đề & Lời mở đầu
**[Nội dung Slide]**
*   **Tên đề tài:** Phân đoạn U não 3D đa lớp sử dụng kiến trúc lai KAN-Attention U-Net
*   **Người trình bày:** [Tên của bạn]
*   **Hội đồng chấm thi:** ...

**[Lời thoại]**
"Kính chào quý thầy cô trong Hội đồng. Hôm nay, em xin phép được trình bày về đề tài nghiên cứu của mình: *Phân đoạn Khối u não 3D đa lớp sử dụng kiến trúc lai KAN-Attention U-Net*. Đây là một giải pháp AI y tế được tinh chỉnh chuyên sâu để đối phó với sự phức tạp của tập dữ liệu BraTS, nhằm bóc tách chính xác 3 vùng bệnh lý: Lõi hoại tử, Phù nề, và U hoạt hóa."

---

## Slide 2: Khó khăn của Bài toán Y khoa (BraTS)
**[Nội dung Slide]**
*   **Mất cân bằng lớp cực độ:** Não khỏe mạnh (>98%) vs Khối u (<2%).
*   **Ranh giới mờ (Fuzzy Boundaries):** Sự lồng ghép phức tạp giữa mô bệnh và mô khỏe.
*   **Giới hạn phần cứng:** Dữ liệu 3D tốn tài nguyên bộ nhớ.

**[Lời thoại]**
"Để giải quyết bài toán BraTS, chúng ta phải đối mặt với thách thức khổng lồ về dữ liệu. Khối u chỉ chiếm chưa tới 2% thể tích hộp sọ, nhưng ranh giới của chúng lại cực kỳ mờ nhạt và lồng ghép hỗn loạn vào nhau. Những kiến trúc U-Net truyền thống hoặc CNN đơn thuần thường bị 'nhiễu' và chẩn đoán nhầm (False Positive) rất nhiều khi gặp các ranh giới mờ này. Để giải quyết trọn vẹn, em đề xuất một kiến trúc lai (Hybrid) gồm 5 trụ cột cốt lõi."

---

## Slide 3: Trụ cột 1 & 2 - Nhánh Nông (CNN) & Nhánh Sâu (SegFormer)
**[Nội dung Slide]**
*(Hình ảnh luồng dữ liệu chia 2 nhánh ngay từ đầu)*
1.  **Shallow Feature Branch (CNN):** 
    * Tổ hợp Conv2d + Residual Block. 
    * Giữ nguyên kích thước 224x224 để bảo tồn 100% chi tiết viền (Edge features).
2.  **Segformer Encoder (Transformer):** 
    * Nén nhanh xuống 7x7. Trích xuất ngữ nghĩa toàn cục (Global Context).

**[Lời thoại]**
"Ngay từ cửa ngõ, dữ liệu ảnh 224x224 được chẻ làm 2 luồng song song. 
Luồng thứ nhất đi vào nhánh **Segformer**. Nhờ cơ chế Transformer, nó nén ảnh rất nhanh xuống mức 7x7 để bắt được 'Ngữ nghĩa toàn cục' – giúp mạng hiểu được khối u đang nằm ở bán cầu não nào. Tuy nhiên, nhược điểm của Transformer là làm mất các chi tiết viền sắc nét.
Để bù đắp, luồng thứ hai đi vào nhánh **Shallow CNN**. Nhánh này sử dụng tổ hợp Conv2d kết hợp Residual Block để giữ nguyên độ phân giải 224x224. Cơ chế Residual đóng vai trò như một bộ giảm xóc, bảo tồn vẹn nguyên các đường nét viền mỏng manh nhất để lát nữa gửi thẳng về cuối mạng làm bản phác thảo chắp vá."

---

## Slide 4: Trụ cột 3 - Nút thắt cổ chai (Tại sao KAN đánh bại MLP?)
**[Nội dung Slide]**
*   **Vị trí Bottleneck:** Kích thước siêu nén 7x7, mang 512 kênh (Khái niệm).
*   **MLP (Truyền thống):** Dùng ReLU cố định ở Nút $\rightarrow$ Cồng kềnh, dễ học vẹt.
*   **KAN (Kolmogorov-Arnold Network):** Dùng B-Splines ở Cạnh $\rightarrow$ Biểu diễn phi tuyến tính mạnh mẽ, bóc tách ranh giới mượt mà.

**[Lời thoại]**
"Khi bức ảnh đi xuống đáy sâu nhất của mạng (Bottleneck), nó bị nén cực đại xuống kích thước 7x7 với 512 kênh. Tại đây, đặc trưng của khối u bị dính chặt vào nhau. 
Thông thường, các nghiên cứu dùng mạng MLP (Multilayer Perceptron) ở đây. Nhưng MLP đặt hàm kích hoạt cứng (như ReLU) ở các Nút. Việc này giống như dùng hàng ngàn que diêm thẳng để ghép thành một hình tròn, nó đòi hỏi ma trận tham số khổng lồ và rất dễ Overfitting.
Điểm nhấn học thuật của em là thay thế MLP bằng **mạng KAN**. KAN loại bỏ hàm ở Nút, mà đặt các hàm đa thức B-Splines có thể học được trực tiếp lên các Cạnh. Thay vì chắp vá các đường thẳng, KAN giống như một sợi dây thép dẻo, tự uốn nắn thành phương trình đường cong hoàn hảo để phân tách các ranh giới tế bào mờ nhạt nhất. KAN mang lại sức mạnh biểu diễn vượt trội với ít tham số hơn."

---

## Slide 5: Trụ cột 4 - Attention Gate & Skip Connection
**[Nội dung Slide]**
*   **Vấn đề U-Net gốc:** Skip Connection bơm cả rác (nhiễu) sang Decoder.
*   **Giải pháp Attention Gate:** "Màng lọc RO" trước khi ghép nối.
    *   Tín hiệu định hướng $g$ (từ dưới lên) kết hợp đặc trưng $x$ (từ ngang sang).
    *   Dập tắt não khỏe mạnh, làm rực sáng khối u.

**[Lời thoại]**
"Khi mạng bắt đầu giải mã và phóng to ảnh lên, nó cần lấy lại nét từ Encoder thông qua Skip Connection. Ở U-Net truyền thống, Skip connection bơm thẳng toàn bộ dữ liệu (bao gồm cả não khỏe mạnh và rác) sang Decoder, làm mạng bị nhiễu.
Do đó, em tích hợp **Attention Gates** đứng chặn tại cầu nối. Lớp Attention này lấy tín hiệu $g$ (đã biết vị trí u) từ dưới lên, áp vào tín hiệu $x$ (nhiều chi tiết viền) từ ngang sang. Nó sinh ra một mặt nạ tự động nhân với 0 để dập tắt toàn bộ các vùng não khỏe mạnh, và nhân với 1 để giữ lại viền khối u. Đặc trưng sau khi được 'lọc sạch' mới được phép ghép nối (Concat) với Decoder."

---

## Slide 6: Trụ cột 5 - Deep Supervision (Giám sát sâu)
**[Nội dung Slide]**
*   **Cơ chế:** Gắn "Trạm kiểm tra" tính Loss tại mọi độ phân giải (14x14, 28x28, 56x56).
*   **Lợi ích:**
    1. Tiêm đạo hàm trực tiếp $\rightarrow$ Trị dứt điểm Vanishing Gradient.
    2. Ép mô hình học ngữ nghĩa cốt lõi, chống học vẹt pixel.
    3. Tự động vô hiệu hóa lúc Inference (Tốc độ không đổi).

**[Lời thoại]**
"Một mạng sâu thường xuyên gặp bệnh 'mờ đạo hàm' (Vanishing Gradient), khiến các lớp dưới đáy học rất chậm. Em giải quyết triệt để bằng kỹ thuật **Deep Supervision**. 
Thay vì đợi đến tấm ảnh 224x224 cuối cùng mới chấm điểm, em gắn các nhánh phụ để ép mạng phải phác thảo và tính Loss khối u ngay từ lúc kích thước chỉ có 14x14 hay 28x28. Nó giống như việc giáo viên kiểm tra và sửa sai học sinh ở từng nét vẽ phác thảo. 
Nhờ vậy, đạo hàm được 'tiêm' trực tiếp vào giữa mạng. Mô hình không những hội tụ nhanh hơn gấp nhiều lần, mà còn bị ép phải hiểu bản chất hình khối của u não thay vì học vẹt. Đặc biệt, các nhánh phụ này chỉ bật lúc Huấn luyện, nên tốc độ dự đoán thực tế cho bệnh nhân không hề bị ảnh hưởng."

---

## Slide 7: Chốt hạ Output (1x1 Conv) & Hậu kỳ (3D CCA)
**[Nội dung Slide]**
*   **1x1 Conv:** Chuyển 16 kênh $\rightarrow$ 4 Logits (Phân loại độc lập từng Pixel).
*   **3D CCA (Connected Component Analysis):** 
    * AI phân loại Pixel $\rightarrow$ CCA gom thành Vùng (Regions).
    * Lọc rác < 50 pixels, duy trì cấu trúc sinh học.

**[Lời thoại]**
"Tại cửa ra cuối cùng, lớp Conv2d 1x1 đóng vai trò như 'vị thẩm phán'. Nó không nhìn không gian xung quanh nữa, mà đâm xuyên qua 16 kênh bằng chứng để xuất ra đúng 4 điểm số xác suất (Logits) độc lập cho từng điểm ảnh.
Tuy nhiên, bản chất AI đánh giá độc lập từng pixel nên đôi khi sinh ra các đốm 'ảo giác'. Do đó, bước Hậu kỳ **3D CCA** được áp dụng để gom các điểm ảnh này thành từng cụm vùng. Nếu một cụm vùng có kích thước quá nhỏ (dưới 50 pixel), thuật toán sẽ kết luận đó là rác và xóa bỏ. Điều này đảm bảo kết quả cuối cùng sạch sẽ và đúng chuẩn y khoa."

---

## Slide 8: Thành tựu & Hướng phát triển quốc tế
**[Nội dung Slide]**
*   **Hiệu năng:** Mean Dice: ~84.08% | HD95: ~8.33 trên BraTS2020.
*   **Hướng phát triển (Tương lai):**
    1.  *Explainable AI (XAI):* Dùng AdaptiveAvgPool2d xuất mã ADN 128 chiều $\rightarrow$ Trực quan hóa t-SNE.
    2.  *Tri-planar Ensembling:* Cắt thêm mặt phẳng Coronal & Sagittal.

**[Lời thoại]**
"Tổng kết lại, kiến trúc đề xuất đã xuất sắc đạt mức Dice Score hơn 84% và tối ưu hóa mạnh mẽ chỉ số HD95. 
Để hướng tới mục tiêu phát hành bài báo khoa học quốc tế trong tương lai, mã nguồn của em đã tích hợp sẵn trạm trích xuất Prototype Features (bằng lớp AdaptiveAvgPool2d ép về vector 128 chiều). Vector này sẽ được dùng cho thuật toán t-SNE để chứng minh trực quan khả năng gom cụm thông minh của AI (XAI). Đồng thời, việc mở rộng dự đoán trên cả 3 mặt phẳng (Tri-planar Ensemble) sẽ là bước đi chiến lược tiếp theo để đưa độ chính xác của dự án lên mức tiệm cận hoàn hảo.
Em xin chân thành cảm ơn quý Hội đồng đã lắng nghe!"
