# Báo Cáo Phân Tích Chuyên Sâu: KANAttentionUNet3D cho Phân Đoạn U Não (BraTS)

Báo cáo này trình bày chi tiết về cơ sở lý thuyết, kiến trúc mạng, cơ chế hoạt động và các kĩ thuật tối ưu được áp dụng trong dự án **KANAttentionUNet3D**. Đây là một hệ thống AI y tế tiên tiến được thiết kế để giải quyết bài toán phân đoạn khối u não đa lớp (Multi-class Brain Tumor Segmentation) từ ảnh cộng hưởng từ (MRI).

---

## Chương 1: Giới thiệu Ứng dụng & Thách thức Y khoa

### 1.1 Mục tiêu Ứng dụng
Hệ thống được xây dựng để tự động phân tích và bóc tách các vùng tổn thương của khối u não dựa trên dữ liệu MRI. Cụ thể, mô hình cần phân loại từng pixel (pixel-wise classification) thành 4 nhóm:
1.  **Lớp 0 (Background):** Vùng nền / Não khỏe mạnh.
2.  **Lớp 1 (NCR/NET - Necrotic and Non-Enhancing Tumor):** Lõi hoại tử của u.
3.  **Lớp 2 (ED - Peritumoral Edema):** Vùng phù nề quanh u.
4.  **Lớp 3 (ET - Enhancing Tumor):** Vùng u đang hoạt hóa / tăng quang.

Từ 4 nhãn gốc này, y học định nghĩa ra 3 vùng lâm sàng chính để đánh giá:
*   **Whole Tumor (WT = NCR + ED + ET):** Toàn bộ khối u.
*   **Tumor Core (TC = NCR + ET):** Lõi khối u.
*   **Enhancing Tumor (ET):** Vùng u hoạt hóa (chỉ riêng lớp 3).

### 1.2 Thách thức của bài toán BraTS
*   **Mất cân bằng dữ liệu cực độ (Extreme Class Imbalance):** Vùng nền (Background) thường chiếm >95% thể tích ảnh, trong khi các vùng u (đặc biệt là ET) chỉ chiếm 1-2%.
*   **Ranh giới mờ (Fuzzy Boundaries):** Khối u não không có ranh giới rõ ràng, dễ bị hòa lẫn với các mô não khỏe mạnh, đặc biệt là vùng phù nề (ED).
*   **Hình thái phức tạp:** Khối u có thể xuất hiện ở bất kỳ vị trí nào, kích thước và hình dạng hoàn toàn ngẫu nhiên.
*   **Giới hạn phần cứng:** Ảnh y tế 3D (Volumetric) cực kì tốn bộ nhớ (VRAM). Xử lý toàn bộ khối 3D độ phân giải cao thường gây lỗi Out of Memory (OOM).

---

## Chương 2: Tiền xử lý & Kỹ thuật Không gian 2.5D

Để giải quyết bài toán bộ nhớ mà vẫn giữ được thông tin không gian (spatial context), dự án áp dụng kĩ thuật **2.5D Volumetric**.

### 2.1 Cơ chế Không gian 2.5D
*   **Lý thuyết:** Thay vì dùng ảnh 2D đơn lẻ (mất thông tin chiều sâu) hoặc 3D toàn phần (quá nặng), cách tiếp cận 2.5D sẽ lấy lát cắt hiện tại $z$ làm trung tâm, kèm theo $k$ lát cắt phía trên và $k$ lát cắt phía dưới để tạo thành một "chồng" (stack).
*   **Triển khai:** Với $k = 2$ và 4 kênh ảnh (T1, T1CE, T2, FLAIR), số lượng kênh đầu vào (in_channels) sẽ là: `4 modalities * (2*2 + 1) = 20 kênh`.
*   **Lợi ích:** Cung cấp cho mạng khả năng "nhìn" được bối cảnh 3D cục bộ (local 3D context) quanh mặt cắt, giúp mô hình phân biệt tốt hơn các cấu trúc kéo dài qua nhiều lát cắt.

### 2.2 Tối ưu hóa Bộ nhớ (Memory Management)
*   **Memmap:** Sử dụng `numpy.memmap` (Memory-mapped file). Thay vì tải toàn bộ file MRI nặng hàng GB lên RAM, `memmap` ánh xạ dữ liệu trực tiếp từ ổ cứng. RAM chỉ tải các khối dữ liệu nhỏ khi mô hình yêu cầu (Lazy Loading).
*   **LRU Cache (Least Recently Used):** Kết hợp `OrderedDict` để cache lại các file vừa được đọc, giúp tăng tốc độ nạp Batch (Dataloader) mà không làm tràn bộ nhớ.
*   **Brain ROI Cropping:** Sử dụng thuật toán Z-Score (Mean + 0.5*Std) để tìm ra hộp giới hạn (Bounding Box) chứa bộ não. Cắt bỏ hoàn toàn các viền đen vô nghĩa xung quanh, giảm đáng kể kích thước tensor truyền vào mạng.

### 2.3 Cân bằng Dữ liệu (Class-Aware Oversampling)
*   Sử dụng `BalancedBatchSampler`. Không lấy mẫu ngẫu nhiên mà ép cấu trúc Batch phải chứa tỷ lệ khối u nhất định để mô hình "học" được vùng u.
*   **Cơ chế phân bổ:** Ép `40%` số lượng lát cắt trong một batch buộc phải chứa khối u (`tumor_ratio=0.4`). Trong 40% khối lượng u này, thuật toán ưu tiên phân bổ: `30% u hoạt hóa (ET)`, `40% lõi u (TC)`, và `30% u toàn phần (WT)`.

---

## Chương 3: Kiến trúc Cốt lõi (KANAttentionUNet3D)

Mô hình là sự lai tạo (Hybrid) giữa **Transformer (SegFormer)**, **Mạng học máy hàm số (KAN - Kolmogorov-Arnold Networks)**, và **U-Net** cổ điển với cơ chế **Attention**.

### 3.1 Shallow Feature Branch (Nhánh Trích xuất Viền Nông)
*   **Lý thuyết:** Các mô hình Transformer (như SegFormer) rất giỏi nắm bắt thông tin tổng thể (Global Context) nhưng khi chia ảnh thành các Patch, chúng thường làm mất đi các chi tiết viền (Edge/High-frequency details) cực kỳ quan trọng trong ảnh y khoa.
*   **Cơ chế:** Nhánh này là một mạng CNN song song, nhận trực tiếp đầu vào gốc. Nó đi qua các lớp Convolution và `ResidualBlock` để tạo ra:
    *   `shallow0`: Đặc trưng viền giữ nguyên kích thước gốc (100% resolution).
    *   `shallow1`: Đặc trưng viền bị giảm 1/2 kích thước.
*   **Ứng dụng:** Hai khối dữ liệu này được "bơm" trực tiếp (skip connection) vào 2 tầng cuối cùng của Bộ giải mã (`d1` và `d0`). Quá trình này cung cấp "bản vẽ chi tiết ranh giới" giúp Decoder sắc nét hóa đường mép của khối u, khắc phục nhược điểm "răng cưa" của Transformer.

### 3.2 Encoder: SegFormer (MiT-B5)
*   **Lý thuyết:** Sử dụng Mix Transformer Encoder (MiT) đã được Pre-trained. SegFormer loại bỏ Positional Encoding (PE) cứng nhắc (như trong ViT) và sử dụng **Mix-FFN** (hàm Conv 3x3) để rò rỉ vị trí qua kĩ thuật zero-padding. Điều này giúp mạng cực kỳ linh hoạt với mọi kích thước ảnh MRI đầu vào.
*   **Dynamic Channel Scaling:** Trọng số Pre-trained của MiT vốn được huấn luyện trên ảnh RGB (3 kênh). Để tiếp nhận đầu vào 20 kênh (từ khối 2.5D), mô hình áp dụng thuật toán chia trung bình trọng số (`scale_factor = 3.0 / in_channels`). Nó sao chép và chia nhỏ trọng số của 3 kênh gốc cho 20 kênh mới, giúp tránh hiện tượng "bùng nổ tín hiệu" (Activation Explosion) ngay tại lớp đầu tiên.

### 3.3 KAN Bottleneck (Kolmogorov-Arnold Network)
*   **Lý thuyết:** Bottleneck là nơi mô hình nén thông tin ở mức sâu nhất (độ phân giải thấp, số kênh cao - ví dụ 512 channels ở 5x5). Thay vì dùng CNN hay Linear layer truyền thống, dự án sử dụng **KANLinear**.
*   **Cơ chế:** Dựa trên định lý biểu diễn Kolmogorov-Arnold. Thay vì đặt các hàm kích hoạt (Activation functions) ở các Node, KAN đặt các hàm B-Splines (hàm đa thức bậc cao) có thể học được trực tiếp lên các Cạnh (Edges).
*   **Ứng dụng:** Lõi khối u thường có hình thái phi tuyến tính cực kì phức tạp (nhiều nang, hoại tử lồi lõm). KAN cung cấp khả năng biểu diễn (Expressivity) mạnh mẽ hơn gấp nhiều lần CNN để "giải mã" các mãnh vỡ không gian phức tạp này. `grid_size=5` và `spline_order=3` đảm bảo độ mịn của đường cong đa thức.

### 3.4 Decoder & Attention Block
*   Bộ giải mã (Decoder) nhận các Feature Map từ Encoder và từ từ khôi phục (Upsample) lại kích thước ảnh ban đầu.
*   **Attention Block:** Tích hợp tại các nút giao (Skip Connection) giữa Encoder và Decoder.
    *   **Cơ chế:** Attention Gate nhận vào 2 luồng dữ liệu: Luồng từ dưới lên (Gating Signal $g$ - chứa Context ngữ nghĩa) và luồng từ ngang sang (Feature $x$ - chứa Chi tiết không gian). Hàm sigmoid sẽ tạo ra một mặt nạ Attention (Attention Mask) giá trị từ 0 đến 1.
    *   **Tác dụng:** Mặt nạ này nhân ngược lại với $x$ (Phép toán $x * psi$). Nó đóng vai trò như một bộ lọc (Filter), tự động làm mờ (trọng số gần 0) các vùng mô não khỏe mạnh / nhiễu, và làm nổi bật (trọng số gần 1) các khu vực tình nghi là khối u.
*   **Residual Block:** Mọi bước giải mã đều đi qua Residual Block (có kết nối tắt) nhằm bảo tồn Gradient, chống hiện tượng suy biến tín hiệu trong quá trình Upsample.

### 3.5 Deep Supervision (Giám sát Sâu)
*   Mô hình xuất ra dự đoán không chỉ ở lớp cuối cùng mà còn ở các tầng giải mã nông hơn (`aux_head4`, `aux_head3`, `aux_head2`).
*   **Tác dụng:** Cung cấp Loss trực tiếp cho các tầng ẩn, ép mạng phải học cách hình thành khối u ngay từ những độ phân giải thấp. Việc này đóng vai trò như một cơ chế tinh chỉnh (Regularization) chống Overfitting và giúp Gradient truyền về Encoder tốt hơn.

---

## Chương 4: Chiến Lược Hàm Loss & Huấn Luyện

### 4.1 Tổ hợp Loss SOTA (3-in-1 SOTAMultiClassLoss)
Hệ thống sử dụng một tổ hợp 3 hàm Loss cực kỳ tinh vi:
1.  **Cross Entropy Loss (Weight = 0.25):** Đóng vai trò là nền tảng (Base). Giám sát sự phân bố xác suất toàn cục, giúp đường cong học tập ổn định, tránh việc mô hình sụp đổ ở các Epoch đầu.
2.  **MultiClass Focal Tversky Loss (Weight = 0.55):** Khắc phục triệt để bài toán Imbalance (Mất cân bằng dữ liệu).
    *   Sử dụng $\alpha = 0.7$ (Phạt False Negative - Bỏ sót u) và $\beta = 0.3$ (Phạt False Positive - Đoán nhầm u). Trọng số này cố tình ép mô hình thà "bắt nhầm còn hơn bỏ sót", cực kỳ quan trọng trong y khoa.
    *   Tham số $\gamma = 0.75$ (Focal) ép mạng dồn sự chú ý (Gradient) vào những pixel "khó" (Hard examples) nằm chênh vênh ở vùng ranh giới.
3.  **MultiClass Boundary Loss (Weight = 0.20):** Thay vì đo lường sự trùng khớp của diện tích, hàm này tính toán khoảng cách (Distance Map) giữa 2 đường viền. Ép mô hình uốn nắn đường mép khối u bám sát y hệt Ground Truth.

### 4.2 Progressive Unfreezing & Mixed Precision
*   **Progressive Unfreezing:** Trong 5 Epoch đầu (`unfreeze_epoch=5`), mạng Encoder SegFormer bị "đóng băng" (requires_grad = False). Chỉ có Decoder và KAN được huấn luyện. Điều này giúp bộ giải mã ngẫu nhiên tự "khởi động" mà không làm hỏng tri thức (Pre-trained weights) quý giá của Encoder. Sau 5 Epoch, mở băng với một `encoder_lr` rất nhỏ.
*   **AMP (Automatic Mixed Precision):** Sử dụng `torch.cuda.amp.GradScaler`. Quá trình Forward Pass chạy ở độ chính xác nửa (FP16) để tính toán ma trận siêu nhanh và tiết kiệm 50% VRAM. Quá trình cập nhật trọng số (Weight Update) chạy ở FP32 để giữ độ chính xác của Gradient.

---

## Chương 5: Xử lý Hậu kỳ (Post-Processing)

Thuật toán ép mạng "bắt nhầm" (nhờ Tversky Loss) sinh ra một tác dụng phụ là tạo ra các đốm nhiễu nhỏ (Ảo giác - Hallucination) rải rác trên mô não. Để giải quyết, quy trình dọn rác 2 bước được áp dụng:

### 5.1 Test-Time Augmentation (TTA)
*   **Cơ chế:** Khi Inference (dự đoán thực tế), ảnh đầu vào được quét qua 4 lần với các phép biến đổi không gian (Gốc, Lật Ngang, Lật Dọc, Lật Ngang+Dọc).
*   **Tác dụng:** Lấy trung bình cộng (Ensemble) ma trận Logits của 4 lần quét. Quá trình này giúp "bào nhẵn" (Smooth) sự bất định của mô hình, triệt tiêu các đốm nhiễu ngẫu nhiên và củng cố niềm tin cho các dự đoán chuẩn xác.

### 5.2 Phân tích Thành phần Liên thông 3D (3D CCA)
*   **Lý thuyết:** Một khối u thực tế phải là một khối mô liên kết chặt chẽ với nhau, không thể là các chấm li ti lơ lửng cách xa khối chính.
*   **Cơ chế:** Hàm `brats_post_processing_cca_3d` thuật toán CCA 3D cô lập các cụm pixel (Cluster). Bất kỳ cụm nào có số lượng Pixel (Volume) nhỏ hơn ngưỡng y khoa cho phép sẽ bị xóa trắng về Background.
*   **Ngưỡng áp dụng:**
    *   **Toàn bộ u (WT) < 300 pixels:** Xóa bỏ.
    *   **Lõi u (TC) < 200 pixels:** Xóa bỏ.
    *   **U hoạt hóa (ET) < 50 pixels:** Xóa bỏ.
    *   Các ngưỡng này được cấu hình kỹ lưỡng dựa trên thực tế lâm sàng của tập BraTS. Kết quả cuối cùng là một bản đồ khối u sắc nét, sạch nhiễu và bám sát sinh học.
